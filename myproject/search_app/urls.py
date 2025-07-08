from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('search/', views.search_api, name='search_api'),
    path('search', views.search_api, name='search_api_no_slash'),  # 支持不带斜杠的请求
] 