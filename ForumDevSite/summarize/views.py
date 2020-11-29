from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .computeSummary import compute
import bs4 as BeautifulSoup
import urllib.request
from .models import Summary
from datetime import datetime

def getUrlText(url) :
    """
    A method that extracts text from a webpage. 
    """
    try :
        # Open and get the paragraph tag text off the webpage specified
        fetched_data = urllib.request.urlopen(url)
        article_read = fetched_data.read()
        article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')
        paragraphs = article_parsed.find_all('p')
        article_content = ''
        
        # Concatenate the different paragraphs
        for p in paragraphs:
            article_content += p.text

    except :
        return ""

    return article_content

def home(request):

    if request.method == 'POST' : # If a submit has been triggered

        is_auth = request.user.is_authenticated # flag used to check whether the user has an account

        if request.POST['action'] == 'save' : # If the user wants to save the summary
            try :
                # Give it the title they asked for or a default
                ttl = request.POST.get('sumTitle')
                update_sum = Summary.objects.get(title="temp"+str(request.user.username))
                dt = datetime.now()
                if ttl == "" or ttl == None :
                    ttl = "Untitled " + str(dt)
                update_sum.title = ttl
                update_sum.save()
            except :
                pass

            return render(request, 'summarize/summarizehome.html', {'submitted': 5})

        elif request.POST['action'] == 'discard' : # If the user doesn't want to keep the summary
            update_sum = Summary.objects.get(title="temp"+str(request.user.username))
            update_sum.delete() # Delete the temporarily saved one

            return render(request, 'summarize/summarizehome.html', {'submitted': 6})

        elif request.POST['action'] == "RawText" : # If the user has pasted in text to be summarized
            text = request.POST.get('usertext')

            if len(text) < 500 : # Too short to get a summary
                context = {'submitted': 2, 'summary': ""}
                return render(request, 'summarize/summarizehome.html', context) # Return error message

            elif len(text) > 20000 and not is_auth : # The author needs to have an account to get summary on text over 20000 characters
                return render(request, 'summarize/summarizehome.html', {'submitted': 4, 'summary': ""})

            smry = compute(text)
            if len(smry) == 0 :
                context = {'submitted': 2, 'summary': ""} # Error has occured
            else :
                context= {'submitted': 1, 'summary': smry}
                if is_auth : # Temporarily save the summary to make it easier to access if user wants to save it later on
                    temp_text = ''
                    for s in smry :
                        temp_text += s + " "
                    summary = Summary(title='temp'+str(request.user.username), content=temp_text, author=request.user)
                    summary.save()
            return render(request, 'summarize/summarizehome.html', context) # All went well, return summary

        elif request.POST['action'] == "File" : # If the user wants a file to be summarized
            try :
                up_file = request.FILES["document"] # Try to get the text from the file
                text = ''
                for line in up_file :
                    text += line.decode()
                
                # If the text in the file is valid, do the same steps/checks as above

                if len(text) < 500 : # Text is too short to summarize
                    context = {'submitted': 2, 'summary': ""}
                    return render(request, 'summarize/summarizehome.html', context)

                elif len(text) > 20000 and not is_auth :
                    return render(request, 'summarize/summarizehome.html', {'submitted': 4, 'summary': ""})
                
                smry = compute(text)
            except :
                # If the file was not compatible, return error message
                smry = ""

            if len(smry) == 0 :
                context = {'submitted': 2, 'summary': ""} # Error message
            else :
                context= {'submitted': 1, 'summary': smry}
                if is_auth :
                    temp_text = ''
                    for s in smry :
                        temp_text += s + " "
                    summary = Summary(title='temp'+str(request.user.username), content=temp_text, author=request.user)
                    summary.save()

            return render(request, 'summarize/summarizehome.html', context) # All good, return summary

        elif request.POST['action'] == 'URL' : # If the user wants a webpage summarized
            url = request.POST.get('urltext')
            text = getUrlText(url) # Get text from the webpage
            
            if len(text) < 500 : # Too short to summarize
                context = {'submitted': 2, 'summary': ""}
                return render(request, 'summarize/summarizehome.html', context) # Give error message

            # Proceed with same steps/checks as per the first one

            elif len(text) > 20000 and not is_auth :
                return render(request, 'summarize/summarizehome.html', {'submitted': 4, 'summary': ""})
            
            smry = compute(text)
            if len(smry) == 0 :                                                                                                                                                                                                    context = {'submitted': 2, 'summary': ""}
            else :
                context= {'submitted': 1, 'summary': smry}
                if is_auth :
                    temp_text = ''
                    for s in smry :
                        temp_text += s + " "
                    summary = Summary(title='temp'+str(request.user.username), content=temp_text, author=request.user)
                    summary.save()

            return render(request, 'summarize/summarizehome.html', context) # All good, return the summary

    else : # If a post hasn't been triggered yet, render the default screen
        context = {'submitted': 3, 'summary': ""}
        return render(request, 'summarize/summarizehome.html', context)


