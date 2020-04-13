from generator import GeneratText
from flask import Flask,request,jsonify,make_response
import logging
import config 

logging.basicConfig(filename=config.LOG_FILE_NAME,
					level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filemode='w',
                    )


app = Flask(__name__)
GT  = GeneratText()

@app.route("/")
def welcomeMessage():

	try:
		return make_response("WelCome To GPT based text generator", 201)
	except Exception as e:
		logging.debug(e)
		

@app.route("/textgen",methods=["POST"])
def  texGenerator():

	try:
		if not request.json or not "text" in request.json:
			return make_response(jsonify({'error': 'not valide post request'}), 400)

		generated_text,status= GT.generateSequences(request.json["text"])

		if status:
			return make_response(jsonify({"original_text":request.json["text"],\
											"generated_text":generated_text}), 201)
		else:
			return make_response(jsonify({"original_text":request.json["text"],\
										"generated_text":"Internal Server error"}), 500)

	except Exception as e:
		logging.debug(e)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404) 


if __name__ == '__main__':
	app.run(host='0.0.0.0',port=6001,debug=True)