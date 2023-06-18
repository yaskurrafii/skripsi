from rest_framework.routers import SimpleRouter
from . import views
from django.urls import path

urlpatterns = [path("test", view=views.test)]

router = SimpleRouter()
router.register("", viewset=views.Skripsi, basename="skripsi")
urlpatterns += router.urls
