"""mlproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from myapp import views

from myapp.views import index
from myapp.views import fileupload
from myapp.views import prediction
from myapp.views import gallery
from myapp.views import contact
from myapp.views import form

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^index/$',index,name='index'),
    url(r'^fileupload/$',fileupload,name='fileupload'),
    url(r'^prediction/$',prediction,name='prediction'),
    url(r'^gallery/$',gallery,name='gallery'),
    url(r'^contact/$',contact,name='contact'),
    url(r'^form/$',form),
]