from flask import Flask, request, jsonify
#from prediction_service import predict_category
from data_processing import predict_category

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    #transaction_data = request.json
    #predicted_category = predict_category()
    predicted_category = predict_category()
    return jsonify({'predicted_category': predicted_category[0]}), 200
if __name__ == '__main__':
    app.run(debug=True)