# fuer mehr info lest https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
#layer which our project product matching will use
FROM python:3.6

#create directory with folder "product_matching_app" where out application will run
RUN mkdir -p /product_matching_app/

#specify working directory
WORKDIR /product_matching_app/

#copy all content in current project directory to the created directory "product_matching_app"
COPY . /product_matching_app/

#install dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#train the sbert model
#run hello world in cmd 

#expose port 8080 for using in local browser (docker container is working isolated from the rest of the world, so we need to forward the port 8080 to our local machine)
#now we can be able to forward port 8080 from docker image to our local port with command -p 8080:8080 or some other port like - p 4200:8080
# -p local port : exposed dockerport
EXPOSE 8080

#TimeZone
ENV TZ Europe/Berlin

# or run application with GUI
ENTRYPOINT  ["python", "app.py"]