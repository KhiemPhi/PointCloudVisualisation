from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('update_points/', views.update_points, name='update_points'),
    path('api/points/<str:cls_label>/<str:batch_num>/', views.get_points_json, name='get_points_json'),
    path('point-cloud-files/', views.list_point_cloud_files, name='point_cloud_files'),
    path('point-cloud-files/sample/download/', views.download_point_cloud_sample, name='download_point_cloud_sample'),
    path('point-cloud-files/<str:file_name>/edit/', views.edit_point_cloud_file, name='edit_point_cloud_file'),

]
