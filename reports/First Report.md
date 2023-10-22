# Assignment 1 - Text De-toxification - Report 1

Sergey Golubev, B20-AI-01 student

s.golubev@innopolis.university

# Hypothesis 1: N-gram model

Firstly, I was planning to build an N-gram model as a baseline solution. The intent to implement this kind of model was to simply mark some words as toxic ones, and replace them with non-toxic N-gram endings. However, I left this idea after I discovered the main assignment dataset is too complicated for this approach.

Some entries could be corrected well with the N-grams, e.g., `I didn't fuck him` -> `I didn't screw him`. But others are unlikely to be de-toxified well, because they don't just contain several toxic words, but utilize complex syntactic structures that themselves represent toxicity, or sarcasm. For instance, `I'm going to hit you in all directions, civil and criminal, on all counts` - the N-gram model wouldn't even recognize any toxic smell here.

# Hypothesis 2: Simple Paraphrasing

I decided to look for a more robust approach, and discovered that the problem could be solved with paraphrasing. The text intent will be preserved after high quality paraphrasing. I decided to fine-tune ??? model to paraphrase examples from the dataset. I use the fine-tuned model to generate several paraphrased options for an input, and then choose the one with minimum toxicity. To measure the toxicity level, I introduce a metric of toxicity ???.

## Toxicity Metric

???

# Hypothesis 3: 

# Results

...
