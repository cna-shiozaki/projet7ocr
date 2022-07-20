###########################
#### EXECUTER L'APP #######
###########################
uvicorn api.run:app --reload
ou 
python -m api.run



docker build --tag projet7 .
docker run -d -p 80:80 projet7
docker tag projet7 cnashiozaki/projet7
docker push cnashiozaki/projet7

docker run -dp 80:80 cnashiozaki/projet7