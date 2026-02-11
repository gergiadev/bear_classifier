from fastai.vision.all import *
import gradio as gr

# Carica il modello
learn = load_learner('export.pkl')
categories = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Interfaccia Gradio
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Bear Classifier",
    description="Carica un'immagine di un orso"
)

demo.launch()
