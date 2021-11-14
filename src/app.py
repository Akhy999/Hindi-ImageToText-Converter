################################################ Importing libraries #############################################

# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
import argparse
import codecs
import cv2

import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

import random
import numpy as np 

from flask import Flask,render_template,request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField, FileField
from wtforms.validators import DataRequired,URL, Regexp
from wtforms import ValidationError
from flask import Markup
from numpy import asarray
import os
from flask_uploads import configure_uploads, IMAGES, UploadSet
import cv2
##################################################################################################################
################################################# app config #####################################################

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
app.config['UPLOADED_IMAGES_DEST'] = 'static/uploads/images'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)
#################################################################################################################
############################################# Forms #############################################################

class Home(FlaskForm):
	image = FileField('image')
	submit=SubmitField('Predict')

################################################################################################################
#########################################global variables#######################################################
cdlink=''                    
calink = ''
################################################################################################################
########################################################loading model##########################################
class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/2.jpg'
    fnCorpus = '../data/hindi_vocab.txt'


def infer(model, fnImg):
    "recognize text in image provided by file path"
    print(f'###############################{fnImg}#################################################')
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0].replace(" ", "") + '"')
    # with open("f.txt", "w") as f:
    #     f.write(recognized[0].replace(" ", ""))
    print('Probability:', probability[0])
    return recognized[0].replace(" ", "")

def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w', encoding='UTF-8').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w', encoding='UTF-8').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=False)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        
        global loaded_model
        loaded_model= Model(codecs.open(FilePaths.fnCharList, encoding='utf-8').read(), decoderType, mustRestore=False)
        #infer(loaded_model, FilePaths.fnInfer)
        
        #loaded_model = model
        print("#############################Model loaded######################################################")

main()
###############################################################################################################
#################################### Inference ##########################################

def model(Model):
	url=calink.replace(" ", "\\ ")
	print("#########################################Inference##############################################") 
	print(url)    
	output = infer(Model,url)                                            
	print(output)
	return output

################################################################################################################
######################### views ################################################################################
UPLOAD_PATH = "./static/uploads/images/"#"C:/Users/sured/Desktop/DDP/master/Flaskimplements/src/static/uploads/images/"
REF_PATH = "uploads/images/"


@app.route('/imagetotext',methods=['GET','POST'])
def imagetotext():
	print(cdlink)
	X=model(loaded_model)
	return render_template('home.html',X=X,cdlink=cdlink,output='')
	

@app.route('/',methods=['GET','POST'])
def index():
	form = Home()
	# form2 = Home2()
	global cdlink
	global calink
	X=''
	cdlink = ''
	if form.validate_on_submit():
		filename = images.save(form.image.data)
		print('f'+filename)
		cdlink=filename
		calink=UPLOAD_PATH+filename
		print(cdlink)
		print(cdlink)
		X=model(loaded_model)
	
	print('else part')
	return render_template('home.html',X=X,form=form,cdlink=cdlink)

################################################################################################################

############################serve###############################################################################

if __name__=='__main__':
	app.run(debug=True)

################################################################################################################