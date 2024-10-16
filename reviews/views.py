from django.shortcuts import render
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from .models import Review
model = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model.load_state_dict(torch.load('./model_state.pth', map_location=torch.device('cpu')))

def classify_review(review_text):
    tokenized_input = tokenizer(review_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**tokenized_input)
        probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()
    return probabilities

def rate_review(review_text):
    probabilities = classify_review(review_text)
    positive_prob = probabilities[0][1]
    negative_prob = probabilities[0][0]

    if positive_prob > negative_prob:  
        rating = round((positive_prob * 4) + 6, 1)  #6 до 10
        return rating, 'Positive'
    else:  
        rating = round((1 - negative_prob) * 5, 1)  # 0 до 5
        return rating, 'Negative'


def review_input(request):
    if request.method == 'POST':
        review_text = request.POST.get('review')
        rating, status = rate_review(review_text)
        review = Review(text=review_text, rating=rating, status=status)
        review.save()

        context = {
            'review': review_text,
            'rating': rating,
            'status': status,
        }
        return render(request, 'reviews/review_result.html', context)

    all_reviews = Review.objects.all()
    return render(request, 'reviews/review_form.html', {'reviews': all_reviews})