from django.shortcuts import render
from .forms import UserForm
from .computeSummary import compute
import PyPDF2 as PdfReader # Reads pdf
from fpdf import FPDF # Allows easy writing of pdfs

def getText(doc) :
    fileObj = open(doc, 'rb')
    pdfReader = PdfReader.PdfFileReader(fileObj)
    text = ""
    for i in range(pdfReader.numPages) :
        pageObj = pdfReader.getPage(i)
        text += pageObj.extractText() + " "

    return text

def summaryPage(request):
    if request.method == "POST" :
        form = UserForm(request.POST)
        f = ""
        if form.is_valid() :
            data = form.cleaned_data.get("your_name")
        f = compute(data)
        context= {'form': form, 'your_name': f}
        return render(request, 'summarizer/home.html', context)
    else :
        form = UserForm()
        context = {'form': form, 'your_name': "This is wrong"}
        return render(request, 'summarizer/home.html', context)

def home1(request):
        if (request.method == 'POST'):
            uploaded_file = request.FILES['document']
            txt = getText(uploaded_file)
            f = compute(txt)
            context = {'form': form, "your_name": f}

            return render(request, 'summarize/home.html', context)

