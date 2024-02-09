from flask import Flask, request, jsonify
import PyPDF2
import pandas as pd
import numpy as np
from keras.utils import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer,  AutoModelForSequenceClassification
import torch
from tqdm import tqdm


#retrieve the pdf and extract its text
def pdf_retriever(file_path): 
    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        text = ''

        # Read each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text




#prepare the model
model_path = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_path, 
                                            do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                            output_attentions=False,
                                                            output_hidden_states=True)


#create a vector of that pdf's text 
def create_vector_from_text(tokenizer, model, text, MAX_LEN = 510):
    
    input_ids = tokenizer.encode(
                        text, 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,                           
                )    

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", 
                            truncating="post", padding="post")
    
    # Remove the outer list.
    input_ids = results[0]

    # Create attention masks    
    attention_mask = [int(i>0) for i in input_ids]
    
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # Add an extra dimension for the "batch" (even though there is only one 
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():        
        logits, encoded_layers = model(
                                    input_ids = input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
        
    # Extract the embedding.
    vector = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    vector = vector.detach().cpu().numpy()
    text_vect = np.array(vector)
    text_vect = text_vect.reshape(1, -1)

    return(text_vect)




#get the similarity score
def plagiarism_score(data, vec):
    x = np.array(data.loc[0])
    vector1 = vec
    vector2 = x.reshape(1, -1)
    cosine_sim = cosine_similarity(vector1, vector2)
    for i in range(1,100000):
        x =np.array(df.loc[i])
        vector2 = x.reshape(1, -1)
        cos_sim = cosine_similarity(vector1, vector2)
        cosine_sim = np.append(cosine_sim, cos_sim)
    plagiarism_score = cosine_sim.max()
    return   plagiarism_score 



#check if the text is plagiarised 
def is_plagiarism(similarity_score):

    is_plagiarism = False
    plagiarism_threshold = 0.6
    if(similarity_score >= plagiarism_threshold):
        is_plagiarism = True

    return is_plagiarism


 




# Specify the path to your large CSV file
file_path = 'chunk0.csv'
data = pd.read_csv(file_path)
column_to_drop = ['ug7v899j', "Unnamed: 0"]
df = data.drop(column_to_drop, axis=1)








app = Flask(__name__)





@app.route('/', methods=['GET', 'POST'])
def checker():
    if request.method == 'POST':
        # Retrieve the text from the form submitted by the user
        directory = request.form['directory']
        txt = pdf_retriever(directory)
        #convert text to vector its shape is(768)
        vec = create_vector_from_text(tokenizer, model, txt)

        similarity_score = plagiarism_score(df, vec)
        #print(similarity_score)
        plagiarism_checker = is_plagiarism(similarity_score)
        if plagiarism_checker == True:
            plg = "plagiarized"
        else:
            plg = "not plagiarized"    
        message = "similarity score is : "+str(similarity_score) + " so your file is "+plg
         
    return message




if __name__ == '__main__':
    app.run(debug=True)
