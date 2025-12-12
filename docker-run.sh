#!/bin/bash
# Quick start script for running SOKEGraph in Docker

echo "SCRIPT RUN FROM: $PWD"
ls -a


# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 SOKEGraph Docker Setup${NC}\n"

# Create necessary directories on host
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/uploads data/outputs data/logs external/output

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env found.${NC}"

    if [ -f .env.example ]; then
        echo -e "${YELLOW}Creating .env from .env.example...${NC}"
        cp .env.example .env
        echo -e "${GREEN}✓ .env created from template.${NC}"
    else
        echo -e "${YELLOW}⚠ No .env.example found. Creating a blank .env file...${NC}"
        touch .env
        echo -e "${GREEN}✓ Blank .env created.${NC}"
    fi
fi

# Build and run
echo -e "\n${YELLOW}Building Docker image...${NC}"
docker-compose build

echo -e "\n${YELLOW}Starting SOKEGraph application...${NC}"
docker-compose up -d

# Wait for service to be ready
echo -e "\n${YELLOW}Waiting for application to start...${NC}"
sleep 5

# Check if container is running
if [ "$(docker ps -q -f name=sokegraph-streamlit)" ]; then
    echo -e "\n${GREEN}✓ SOKEGraph is running!${NC}"
    echo -e "\n${BLUE}Access the application at:${NC} http://localhost:8501"
    echo -e "\n${BLUE}Your output files will be available in:${NC}"
    echo "  - ./data/outputs/       (ranking results, graphs)"
    echo "  - ./external/output/    (pipeline outputs)"
    echo "  - ./data/logs/          (application logs)"
    echo -e "\n${BLUE}Useful commands:${NC}"
    echo "  View logs:    docker compose logs -f"
    echo "  Stop:         docker compose down"
    echo "  Restart:      docker compose restart"
    echo "  Shell access: docker exec -it sokegraph-streamlit bash"
else
    echo -e "\n${YELLOW}⚠ Container failed to start. Check logs:${NC}"
    docker-compose logs
fi
