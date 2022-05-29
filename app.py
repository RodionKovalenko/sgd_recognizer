import os
from flask import Flask
from flask import render_template, request
from flask_restful import Api

from src.webservice import sdg_classifier

app = Flask(__name__)
api = Api(app)
sgd_clf = sdg_classifier.Sgd_Classifier()

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_FOLDER, "resources")


@app.route('/')
@app.route('/index')
def start_app():
    recognition_levels = [[1, 'selected', 'Sentence'], [2, '', 'Paragraph'], [3, '','Text']]
    
    return render_template('index.html',
                           sdg_recognizer_result={},
                           sim_threshold=0.6,
                           recognition_levels=recognition_levels,
                           use_gnb_checked = ''
                           )


@app.route('/about')
def about_action():
    return render_template('about.html')

# when Button "Vergleich" is clicked


@app.route("/recognize_sdg/", methods=['GET', 'POST'])
def on_compare_strings_action():
    text_to_recognize = request.args.get('text')[:]
    sim_threshold = request.args.get('sim_threshold')[:]
    use_gnb = request.args.get('use_gnb')
    recognition_level = int(request.args.get('recognition_levels')[:])
    recognition_levels = [[1, '', 'Sentence'], [2, '', 'Paragraph'], [3, '', 'Text']]
    
    if recognition_level == 1:
        recognition_levels[0][1] = 'selected'
    elif recognition_level == 2:
        recognition_levels[1][1] = 'selected'
    elif recognition_level ==3:
         recognition_levels[2][1] = 'selected'
    if text_to_recognize:
        text_to_recognize = text_to_recognize.strip()
       

    sdg_recognizer_result = sgd_clf.predict_by_gnb_and_cos_sim(text_to_recognize,
                                                               True,
                                                               sim_threshold,
                                                               recognition_level,
                                                               use_gnb)
        
    if not sim_threshold or (sim_threshold and float(sim_threshold) > 1):
        sim_threshold = sgd_clf.cos_sim_threshold
    if use_gnb:
        use_gnb = 'checked'
    else:
        use_gnb = ''
  
    return render_template('index.html',
                           sdg_recognizer_result=sdg_recognizer_result,
                           text=text_to_recognize,
                           sim_threshold=sim_threshold,
                           recognition_levels=recognition_levels,
                           use_gnb_checked = use_gnb)


# Sbert API
api.add_resource(sgd_clf, '/api/v1/recognize_sdg', endpoint='sdg_claffier')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4200, debug=False)

























