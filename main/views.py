from rest_framework.viewsets import ModelViewSet
from .models import SkripsiModel
from .serializer import SkripsiSerializer
from rest_framework.response import Response
from .func import predict
from django.http.response import HttpResponse

# Create your views here.


class Skripsi(ModelViewSet):
    queryset = SkripsiModel.objects.all()
    serializer_class = SkripsiSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        media_path = "/".join(serializer.data["image"].split("/")[3:])
        hasil = predict(media_path)
        if hasil.detach().cpu().numpy()[0] == 1:
            data = {"predict": "abnormal"}
        elif hasil.detach().cpu().numpy()[0] == 0:
            data = {"predict": "normal"}
        else:
            data = {"predict": "not Found"}
        return Response(data, headers=headers)


def test(request):
    return HttpResponse("<h1>Ok</h1>")
