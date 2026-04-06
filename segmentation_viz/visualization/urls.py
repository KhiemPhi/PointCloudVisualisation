from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),

    # Point-cloud file management pages
    path('point-cloud-files/', views.list_point_cloud_files, name='point_cloud_files'),
    path('point-cloud-files/sample/download/', views.download_point_cloud_sample, name='download_point_cloud_sample'),
    path('point-cloud-files/<str:file_name>/edit/', views.edit_point_cloud_file, name='edit_point_cloud_file'),

    # API endpoints used by the frontend JavaScript
    path('api/points/<str:cls_label>/<str:batch_num>/', views.get_points_json, name='get_points_json'),
    path('update_points/', views.update_points, name='update_points'),

    # ADDED: endpoint used by the new BITSI buttons in the UI.
    # mode is expected to be either 'single' or 'multi'.
    path('segment/<str:mode>/', views.run_bitsi_segmentation, name='run_bitsi_segmentation'),
]
