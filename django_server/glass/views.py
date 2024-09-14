from django.shortcuts import render, redirect
from django.contrib import messages
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .forms import GlassesForm
from .models import Glasses
from .serializers import GlassesSerializers
from rest_framework.response import Response


def register_glasses(request):
    if request.method == 'POST':
        form = GlassesForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Glass registration successful!')
            return redirect('glasses_landing')
    else:
        form = GlassesForm()

    return render(request, 'register_glasses.html', {'form': form})


@api_view(['GET'])
def glasses_landing(request):
    glasses = Glasses.objects.all()
    serializer = GlassesSerializers(glasses, many=True)
    return render(request, 'glasses_landing.html', {'glasses': serializer.data})


def index(request):
    return render(request , 'index.html')

@api_view(['POST'])
def glass_detail(request):

    try:

        glass_id = request.data.get('glass_id')
        if glass_id:
            glass = Glasses.objects.get(glass_id = glass_id)
            serializer = GlassesSerializers(glass)

            return Response(serializer.data)

    except Glasses.DoesNotExist:
        return Response({"error": "glass does not exist"}, status=400)

    except Exception as e:
        return Response({"error" : str(e)} , status = 500)
