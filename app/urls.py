from django.urls import path
from .views import home, busquedaPdfs, busuedalinks, entrenamoento, predict, resumen2

urlpatterns = [
    path('', home, name="home"),
    path('pdfs/', busquedaPdfs, name="busquedaPdfs"),
    path('links/', busuedalinks, name="busuedalinks"),
    path('ent/', entrenamoento, name="entrenamoento"),
    path('pre/', predict, name="predict"),
    path('res/', resumen2, name="resumen2"),
]
