docker-compose down --rmi all --volumes --remove-orphans
docker image prune -af
docker-compose build --no-cache
docker-compose up