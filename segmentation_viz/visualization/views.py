from django.shortcuts import render
import os, glob, json, math
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

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

        else:
            return HttpResponseBadRequest("Unknown action")
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        
        return JsonResponse({"status": "success", "data": content})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})
