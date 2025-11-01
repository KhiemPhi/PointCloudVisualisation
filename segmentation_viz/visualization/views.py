from django.shortcuts import render
import os, glob, json, math, re
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest, Http404, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils.text import get_valid_filename
from datetime import datetime

POINT_CLOUD_NAME_RE = re.compile(r'^[A-Za-z0-9]+_[0-9]+_points\.json$')


def _is_valid_point_cloud_filename(file_name: str) -> bool:
    return bool(POINT_CLOUD_NAME_RE.fullmatch(file_name))


def get_points_json(request, cls_label, batch_num):
    file_path = os.path.join(settings.BASE_DIR, 'static', f"{cls_label}_{batch_num}_points.json")
    if not os.path.exists(file_path):
        return HttpResponseBadRequest("File not found")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return JsonResponse(data, safe=False)

def index(request):
    static_dir = os.path.join(settings.BASE_DIR, 'static')
    pattern = os.path.join(static_dir, '*_points.json')
    file_paths = glob.glob(pattern)
    
    # Build a mapping: class_name -> list of batch numbers
    mapping = {}
    for fp in file_paths:
        file_name = os.path.basename(fp)  # e.g. airplane_0_points.json
        if file_name.endswith("_points.json"):
            prefix = file_name[:-len("_points.json")]  # e.g. airplane_0
            parts = prefix.split('_')
            if len(parts) >= 2:
                class_name = parts[0]
                batch_num = parts[1]
                if class_name not in mapping:
                    mapping[class_name] = []
                if batch_num not in mapping[class_name]:
                    mapping[class_name].append(batch_num)
    
    # Sort batch numbers numerically.
    for cls in mapping:
        mapping[cls].sort(key=lambda x: int(x))
    
    classes = sorted(mapping.keys())
    mapping_json = json.dumps(mapping)
    
    return render(request, 'visualization/index.html', {
        'classes': classes,
        'mapping_json': mapping_json,
    })

def list_point_cloud_files(request):
    static_dir = os.path.join(settings.BASE_DIR, 'static')
    os.makedirs(static_dir, exist_ok=True)

    upload_error = None
    upload_message = None

    if request.method == 'POST':
        uploaded = request.FILES.get('point_cloud_file')
        if not uploaded:
            upload_error = "Select a JSON file to upload."
        elif not uploaded.name.lower().endswith('.json'):
            upload_error = "Only .json files are supported."
        else:
            safe_name = get_valid_filename(uploaded.name)
            if not safe_name:
                upload_error = "Unable to derive a valid file name."
            else:
                if not safe_name.lower().endswith('.json'):
                    safe_name = f"{safe_name}.json"
                if not _is_valid_point_cloud_filename(safe_name):
                    upload_error = "Filename must follow objname_batch_points.json (e.g. mug_001_points.json)."
                else:
                    dest_path = os.path.join(static_dir, safe_name)
                    file_bytes = b''.join(uploaded.chunks())
                    try:
                        decoded = file_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        upload_error = "Uploaded file must be UTF-8 encoded JSON."
                    else:
                        try:
                            parsed = json.loads(decoded)
                        except json.JSONDecodeError as exc:
                            upload_error = f"Invalid JSON: {exc}"
                        else:
                            with open(dest_path, 'w', encoding='utf-8') as out_file:
                                json.dump(parsed, out_file, indent=2)
                            upload_message = f"Uploaded {safe_name} to the static directory."

    files = []
    static_url_base = settings.STATIC_URL
    if not static_url_base.endswith('/'):
        static_url_base += '/'

    for entry in sorted(os.listdir(static_dir)):
        full_path = os.path.join(static_dir, entry)
        if os.path.isfile(full_path) and entry.lower().endswith('.json'):
            size_kb = round(os.path.getsize(full_path) / 1024, 2)
            modified_dt = datetime.fromtimestamp(os.path.getmtime(full_path))
            files.append({
                "name": entry,
                "size_kb": size_kb,
                "modified": modified_dt,
                "url": f"{static_url_base}{entry}",
            })

    context = {
        "files": files,
        "upload_error": upload_error,
        "upload_message": upload_message,
    }
    return render(request, 'visualization/point_cloud_files.html', context)

def edit_point_cloud_file(request, file_name):
    static_dir = os.path.join(settings.BASE_DIR, 'static')
    safe_name = os.path.basename(file_name)
    if safe_name != file_name or not safe_name.lower().endswith('.json'):
        return HttpResponseBadRequest("Invalid file selection.")
    if not _is_valid_point_cloud_filename(safe_name):
        return HttpResponseBadRequest("Invalid file selection.")

    file_path = os.path.join(static_dir, safe_name)
    if not os.path.isfile(file_path):
        raise Http404("File not found")

    error = None
    success_message = None

    if request.method == 'POST':
        submitted_content = request.POST.get('file_content', '')
        if not submitted_content.strip():
            error = "File content cannot be empty."
            content_for_form = submitted_content
        else:
            try:
                parsed = json.loads(submitted_content)
            except json.JSONDecodeError as exc:
                error = f"Invalid JSON: {exc}"
                content_for_form = submitted_content
            else:
                with open(file_path, 'w', encoding='utf-8') as out_file:
                    json.dump(parsed, out_file, indent=2)
                success_message = f"Saved changes to {safe_name}."
                content_for_form = json.dumps(parsed, indent=2)
    else:
        with open(file_path, 'r', encoding='utf-8') as in_file:
            content_for_form = in_file.read()
        try:
            parsed = json.loads(content_for_form)
            content_for_form = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass

    context = {
        "file_name": safe_name,
        "file_content": content_for_form,
        "error": error,
        "success_message": success_message,
    }
    return render(request, 'visualization/edit_point_cloud_file.html', context)


def download_point_cloud_sample(request):
    sample_payload = [
        {
            "coordinates": [[0.0, 0.0, 0.0]],
            "part_label": [0],
            "cls_label": "object_name"
        }
    ]
    body = json.dumps(sample_payload, indent=2)
    response = HttpResponse(body, content_type='application/json; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename=objname_batchnum_points.json'
    return response


def coordinates_match(coord, x, y, z, tol=1e-6):
    # Compare the first three elements of the coordinate array.
    return (math.isclose(coord[0], x, abs_tol=tol) and 
            math.isclose(coord[1], y, abs_tol=tol) and 
            math.isclose(coord[2], z, abs_tol=tol))

@csrf_exempt  # For demonstration. In production, use proper CSRF handling.
@require_POST
def update_points(request):
    """
    This view updates the original JSON file (e.g. "airplane_0_points.json") which is assumed to have the structure:

    [
      {
         "coordinates": [
            [ x1, y1, z1 ],
            [ x2, y2, z2 ],
            ...,
            [ x_new, y_new, z_new, "custom" ]
         ],
         "part_label": [ label1, label2, ..., -1 ],
         "cls_label": "airplane"
      }
    ]

    Expected payloads:

    For "append":
      {
         "action": "append",
         "data": {
             "cls_label": <str>,
             "batch_num": <str>,
             "x": <float>,
             "y": <float>,
             "z": <float>
         }
      }
      → Appends [x, y, z, "custom"] to "coordinates" and –1 to "part_label".

    For "reassign":
      {
         "action": "reassign",
         "data": {
             "cls_label": <str>,
             "batch_num": <str>,
             "x": <float>,
             "y": <float>,
             "z": <float>,
             "new_part_label": <int>
         }
      }
      → Searches for a custom point (coordinate array with 4 elements with fourth element "custom") that matches
         [x,y,z] (within tolerance) and updates the corresponding "part_label" to the new integer.
         The coordinate remains unchanged (with the "custom" flag preserved).

    For "delete":
      {
         "action": "delete",
         "data": {
             "cls_label": <str>,
             "batch_num": <str>,
             "x": <float>,
             "y": <float>,
             "z": <float>
         }
      }
      → Searches for a custom point with matching [x,y,z] (in an array of 4 elements) and deletes it
         from both "coordinates" and "part_label".
    """
    try:
        payload = json.loads(request.body)
        action = payload.get("action")
        data = payload.get("data")
        
        cls_label = data.get("cls_label")
        batch_num = data.get("batch_num")
        if not cls_label or not batch_num:
            return HttpResponseBadRequest("Missing cls_label or batch_num")
        
        file_path = os.path.join(settings.BASE_DIR, 'static', f"{cls_label}_{batch_num}_points.json")
        if not os.path.exists(file_path):
            return HttpResponseBadRequest("File not found")
        
        with open(file_path, 'r') as f:
            content = json.load(f)
        if not content or not isinstance(content, list):
            return HttpResponseBadRequest("File format invalid")
        point_obj = content[0]
        
        if action == "append":
            x = data.get("x")
            y = data.get("y")
            z = data.get("z")
            if x is None or y is None or z is None:
                return HttpResponseBadRequest("Missing coordinate value")
            point_obj["coordinates"].append([x, y, z, "custom"])
            point_obj["part_label"].append(-1)
            
        elif action == "reassign":
            x = data.get("x")
            y = data.get("y")
            z = data.get("z")
            new_part_label = data.get("new_part_label")
            if x is None or y is None or z is None or new_part_label is None:
                return HttpResponseBadRequest("Missing data for reassign")
            try:
                new_part_label = int(new_part_label)
            except:
                return HttpResponseBadRequest("new_part_label must be an integer")
            found_index = None
            for i, coord in enumerate(point_obj["coordinates"]):
                if len(coord) == 4 and coord[3] == "custom" and coordinates_match(coord, x, y, z):
                    found_index = i
                    break
            if found_index is None:
                return HttpResponseBadRequest("Custom point not found for reassign")
            point_obj["part_label"][found_index] = new_part_label
            # Do not remove the "custom" flag; it should remain to mark it as a custom point.
            
        elif action == "delete":
            x = data.get("x")
            y = data.get("y")
            z = data.get("z")
            if x is None or y is None or z is None:
                return HttpResponseBadRequest("Missing data for delete")
            found_index = None
            for i, coord in enumerate(point_obj["coordinates"]):
                if len(coord) == 4 and coord[3] == "custom" and coordinates_match(coord, x, y, z):
                    found_index = i
                    break
            if found_index is None:
                return HttpResponseBadRequest("Custom point not found for deletion")
            point_obj["coordinates"].pop(found_index)
            point_obj["part_label"].pop(found_index)
            
        elif action == "add_arrow":
            p1 = data.get("p1")
            p2 = data.get("p2")
            if not (p1 and p2):
                return HttpResponseBadRequest("Missing arrow endpoints")
            # ensure the arrows list exists
            if "arrows" not in point_obj:
                point_obj["arrows"] = []
            # append [ [x1,y1,z1], [x2,y2,z2] ]
            point_obj["arrows"].append([
                [p1["x"], p1["y"], p1["z"]],
                [p2["x"], p2["y"], p2["z"]],
            ])
            
        elif action == "delete_arrow":
            p1 = data.get("p1")
            p2 = data.get("p2")
            if not (p1 and p2):
                return HttpResponseBadRequest("Missing arrow endpoints")
            if "arrows" in point_obj:
                for i, arrow in enumerate(point_obj["arrows"]):
                    # match by coordinates
                    if (coordinates_match(arrow[0], p1["x"], p1["y"], p1["z"]) and
                        coordinates_match(arrow[1], p2["x"], p2["y"], p2["z"])):
                        point_obj["arrows"].pop(i)
                        break
                    
        elif action == "add_screw":
            # expects data: x,y,z plus dir_x,dir_y,dir_z each ∈ {-1,0,+1}
            if "screws" not in point_obj:
                point_obj["screws"] = []
            coords = [
                data["x"], data["y"], data["z"],
                data["dir_x"], data["dir_y"], data["dir_z"]
            ]
            if coords not in point_obj["screws"]:
                point_obj["screws"].append(coords)
        
        elif action == "add_segment":
            pts = data.get("points", [])
            coords = [[p["x"], p["y"], p["z"]] for p in pts if "x" in p and "y" in p and "z" in p]
            if "segments" not in point_obj:
                point_obj["segments"] = []
            point_obj["segments"].append(coords)

        elif action == "add_circle" :
            cx = data.get("cx"); cy = data.get("cy")
            cz = data.get("cz"); r = data.get("r")
            if None in (cx, cy, cz, r):
                return HttpResponseBadRequest("Missing circle parameters")
            if "circles" not in point_obj:
                point_obj["circles"] = []
            coords = [cx, cy, cz, r]
            if coords not in point_obj["circles"]:
                point_obj["circles"].append(coords)
        
        elif action == "add_segment_with_circle":
            pts = data.get("points", [])
            cx = data.get("cx")
            cy = data.get("cy")
            cz = data.get("cz")
            nx = data.get("nx")
            ny = data.get("ny")
            nz = data.get("nz")
            r  = data.get("r")

            if None in (cx, cy, cz, r):
                return HttpResponseBadRequest("Missing circle parameters")

            # Add segment
            coords = [[p["x"], p["y"], p["z"]] for p in pts if "x" in p and "y" in p and "z" in p]
            if "segments" not in point_obj:
                point_obj["segments"] = []
            point_obj["segments"].append(coords)

            # Add circle
            if "circles" not in point_obj:
                point_obj["circles"] = []
            circle_data = [cx, cy, cz, nx, ny, nz, r]
            if circle_data not in point_obj["circles"]:
                point_obj["circles"].append(circle_data)

        elif action == "add_handle":
            pts = data.get("points", [])
            handle_index = data.get("handle_index", 0)
            
            if not pts:
                return HttpResponseBadRequest("Missing handle points")
            
            if "handles" not in point_obj:
                point_obj["handles"] = []
            
            handle_coordinates = []
            for i in range(len(pts) - 1):
                handle_coordinates.append([
                    [pts[i]["x"], pts[i]["y"], pts[i]["z"]],
                    [pts[i + 1]["x"], pts[i + 1]["y"], pts[i + 1]["z"]]
                ])
            
            handle_data = {
                "coordinates": handle_coordinates,
                "handle_index": handle_index,
                "total_points": len(pts)
            }
            
            # Add or update handle
            if handle_index < len(point_obj["handles"]):
                point_obj["handles"][handle_index] = handle_data
            else:
                point_obj["handles"].append(handle_data)

        elif action == "add_annotated_segment":
            pts = data.get("points", [])
            annotated_label = data.get("annotated_label", None)
            if annotated_label is None:
                return HttpResponseBadRequest("Missing annotated_label for annotated segment")
            
            coords = [[p["x"], p["y"], p["z"]] for p in pts if "x" in p and "y" in p and "z" in p]
            if "annotated_segments" not in point_obj:
                point_obj["annotated_segments"] = []
            point_obj["annotated_segments"].append({
                "coordinates": coords,
                "annotated_label": annotated_label
            })

        else:
            return HttpResponseBadRequest("Unknown action")
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        
        return JsonResponse({"status": "success", "data": content})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})