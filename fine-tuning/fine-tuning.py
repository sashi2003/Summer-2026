from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This internship situation has been confusing")
print(result)

result2 = classifier("The weather today is okay I guess")
print(result2)

result3 = classifier("I absolutely loved this experience, best decision ever")
print(result3)