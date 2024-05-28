
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL = tf.keras.models.load_model("D:\Smart Farmer\Potato\potato.keras")

# CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
    
#     if predicted_class == "Early Blight":
#         output = {
#             'name' : 'potato',
#             'class': predicted_class,
#             'confidence': float(confidence),
#             'arabic_name': "اللفحه المبكره",
#             "details": {
                
#                 "symptoms": "تحدث الاعراض علي الاشجار القديمه والسيقان والثمار  تظهر بقع ذات لون رمادي مائل الي البني وتكةن بقع محاطه بهاله صفراء  ومع تقدم المرض تكون الورقه باكملها صقراء الناتج عن نقص الكلورفيل وتتساقط الاوراق وتكون الاوراق اكثر عرضه للشمس ويتبعه عفن علي الثمار ",
#                 "cause": "", 
#                 "fight": "استخدام مبيدات الفطريات مثل  الاوزوكسيستروبين والبيرايكلوستيروبين و كما يوصي بالخلط بين المركبات"
#             }
#         }
#     elif predicted_class == "Late Blight":
#         output = {
#             'name' : 'potato',
#             'class': predicted_class,
#             'confidence': float(confidence),
#             'arabic_name': "اللفحه المتاخره",
#             "details": {
#                 "symptoms": "تبدأ أعراض المرض بظهور بقع بنية صغيرة على الأوراق تكبر في الحجم في الظروف الملائمة حتى تعم كل سطح الورقة. وقد تمتد الإصابة إلي أعناق الأوراق ، وعند اشتداد الإصابة يصاب كل المجموع الخضري ويموت النبات ، ويظهر على السطح السفلي للبقع المصابة ( وقليلا على السطح العلوي) نمو زغبي رمادي اللون ، هو عبارة عن حوامل الفطر الجرثومية والأكياس الجرثومية بأعداد كبيرة ويشاهد هذا الزغب بوضوح في الجو الرطب وعند تلبد الجو بالغيوم. وتصبح الدرنات المصابة صغيرة الحجم قليلة العدد ، ويظهر عليها عفن بني في الأنسجة الداخلية تحت القشرة مباشرة، ويمتد هذا العفن بغير انتظام للداخل  وفي الأراضي الثقيلة يلاحظ أن هذا العفن يكون طرياً، وتتلف الدرنات نتيجة لإصابتها بكائنات ثانوية، وقد تتلف بعض هذه الدرنات تماماً قبل ضم المحصول",
#                 "cause": "", 
#                 "fight":"إتباع دورة زراعية يراعي فيها عدم تعاقب بطاطس وطماطم في نفس الحقل. عدم زراعة طماطم في العروة الشتوية بالقرب من زراعة بطاطس. حرق مخلفات البطاطس والطماطم للتخلص من مصدر العدوى الموجود بها وهو الجراثيم البيضية للفطر. أما المقاومة الكيماوية فتتم برش النباتات 4-6 مرات وبين كل رشة والأخرى 15 يوماً بأحد المطهرات الفطرية الموصى بها من قبل وزارة الزراعة"
#             }
#         }
#     else:
#         output = {
#             'name':'potato',
#             'class': predicted_class,
#             'confidence': float(confidence)
            
#         }
#     return output

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(r"D:\Smart Farmer\Tomato\tomato.h5")

CLASS_NAMES = ['Bacterial Spot',
 'Early Blight',
 'Healthy',
 'Late Blight',
 'Leaf Mold',
 'Mosaic Virus',
 'Septoria leaf Spot',
 'Spider Mites Two Spotted ',
 'Target Spot',
 'Yellow Leaf Curl Virus']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'name':'Tomato',
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
   