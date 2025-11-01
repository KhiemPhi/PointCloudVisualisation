from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('update_points/', views.update_points, name='update_points'),
    path('api/points/<str:cls_label>/<str:batch_num>/', views.get_points_json, name='get_points_json'),
    path('point-cloud-files/', views.list_point_cloud_files, name='point_cloud_files'),

]
