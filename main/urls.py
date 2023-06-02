from rest_framework.routers import SimpleRouter
from . import views

urlpatterns = []

router = SimpleRouter()
router.register("", viewset=views.Skripsi, basename="skripsi")
urlpatterns += router.urls
