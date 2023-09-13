.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y DeepFakeDetection || :
	@pip install -e .

install_requirements:
	@pip install -r requirements.txt

#################### BUILD CLOUD API ###################

##Test local api
run_api:
	uvicorn DeepFakeDetection.api.fast:app --reload

## Build docker image
build_docker:
	docker build --no-cache --tag=${GCR_IMAGE} .
#docker images: displays all the images

run_docker:
	docker run -e PORT=8000 -p 8000:8000 ${GCR_IMAGE}

## Image in Google Run Platform

buid_cloud:
	docker build -t ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod .

run_cloud:
	docker run -e PORT=8080 -p 8080:8080 --env-file .env ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

push_cloud:
	docker push ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

deploy_cloud:
	gcloud run deploy --image ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod --memory ${GCR_MEMORY} --region ${GCP_REGION} --env-vars-file .env.yaml
#gcloud run services delete deepfakepreproc #delete from gcloud run

######################## RUN FE ########################
streamlit:
	-@streamlit run app.py
