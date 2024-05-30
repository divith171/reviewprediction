from flask import Flask, render_template, request,Response,render_template_string, jsonify
import os 
import time
import subprocess
import logging
from reviewprediction.pipeline.prediction import PredictionPipeline, CustomData





app = Flask(__name__) # initializing a Flask app
main_py_timestamps = []  # List to store the timestamps when main.py was triggered


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['POST','GET'])  # route to train the pipeline
def training():
        if request.method == 'GET':
         if os.path.exists('artifacts\model_trainer\model.joblib'):
            return render_template_string("""
                <script>
                    if (confirm("Model is already trained. Do you still want to proceed and train the model again?")) {
                        window.location.href = "/info";
                    } else {
                        window.location.href = "/"; // Go back to the home page
                    }
                </script>
            """)
         else:
             
            return render_template('info.html')

            
        else:
            return "Invalid request method."

@app.route('/info')
def info():
    return render_template('info.html')
@app.route('/proceed_training', methods=['GET'])
def proceed_training():
   
    if request.method == 'GET':
        os.system("python main.py")
        
        return "Model trained successfully!"
    else:
        return "Invalid request method."






@app.route('/predict', methods=['POST', 'GET']) # route to show the predictions in a web UI
def predict():
    if os.path.exists('artifacts\model_trainer\model.joblib'):
        try:
            data = CustomData(
                order_item_id=int(request.form.get('order_item_id')),
                price=float(request.form.get('price')),
                freight_value=float(request.form.get('freight_value')),
                product_category_name=str(request.form.get('product_category_name')),
                product_name_length=float(request.form.get('product_name_length')),
                product_description_length=float(request.form.get('product_description_length')),
                product_photos_qty=float(request.form.get('product_photos_qty')),
                product_weight_g=float(request.form.get('product_weight_g')),
                product_length_cm=float(request.form.get('product_length_cm')),
                product_height_cm=float(request.form.get('product_height_cm')),
                product_width_cm=float(request.form.get('product_width_cm')),
                seller_zip_code_prefix=int(request.form.get('seller_zip_code_prefix')),
                seller_city=str(request.form.get('seller_city')),
                seller_state=str(request.form.get('seller_state')),
                order_status=str(request.form.get('order_status')),
                customer_zip_code_prefix=int(request.form.get('customer_zip_code_prefix')),
                customer_city=str(request.form.get('customer_city')),
                customer_state=str(request.form.get('customer_state')),
                review_id=str(request.form.get('review_id')),
                review_comment_title=str(request.form.get('review_comment_title')),
                review_comment_message=str(request.form.get('review_comment_message')),
                review_creation_date=str(request.form.get('review_creation_date')),
                review_answer_timestamp=str(request.form.get('review_answer_timestamp')),
                payment_sequential=int(request.form.get('payment_sequential')),
                payment_type=str(request.form.get('payment_type')),
                payment_installments=int(request.form.get('payment_installments')),
                payment_value=float(request.form.get('payment_value')),
                purchase_delivery_difference=int(request.form.get('purchase-delivery difference')),
                estimated_actual_delivery_difference=int(request.form.get('estimated-actual delivery difference')),
                price_category=str(request.form.get('price_category')),
                purchase_delivery_diff_per_price=float(request.form.get('purchase_delivery_diff_per_price')),
                review_availability=int(request.form.get('review_availability'))
            )
            final_data = data.data_transformer()
            obj = PredictionPipeline()
            pred = obj.predict(final_data)
            sentiment = 'The review is positive!' if pred == 1 else 'The review is negative!'
            return render_template('result.html', prediction=sentiment)
        
        except Exception as e:
            logging.exception("An error occurred while processing the request:")
            return 'Something went wrong while processing the request. Please try again later.'
    else:
        return "Please Train the model and proceed to get your prediction"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
