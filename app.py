import gradio as gr
from fastai.vision.all import *

learn = load_learner('forest.pkl')
labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


title = "Forest Fire Classifier"
description = "A forest fire classifier trained on the dataset gotten from the internet with duckduckgo. A model was trained on the data using Fastai. Created as a demo for Gradio and HuggingFace Spaces."
gr.Interface(fn=predict, #model function
             inputs=gr.inputs.Image(shape=(512, 512)),#specify the input. In our case an image with
             outputs=gr.outputs.Label(num_top_classes=2),
             title=title,
             description=description,
             examples=['hercules.jpg','aerial-top-view-forest-tree-260nw-2033096327.webp'],
             interpretation='default',
             enable_queue=True).launch()