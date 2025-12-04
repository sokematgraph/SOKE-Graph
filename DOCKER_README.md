# 🐳 SOKEGraph Docker Deployment Guide

## Quick Start

### Option 1: Using the Quick Start Script (Easiest)

```bash
# Make script executable
chmod +x docker-run.sh

# Run it
./docker-run.sh
```

This will:
- Create necessary data directories
- Set up .env file
- Build the Docker image
- Start the application
- Show you the URL to access it

### Option 2: Manual Docker Compose

```bash
# 1. Create data directories
mkdir -p data/{uploads,outputs,logs} external/output

# 2. Create .env file (optional, for API keys)
cp .env.example .env
# Edit .env with your API keys

# 3. Build and start
docker-compose up -d

# 4. Access the app at http://localhost:8501
```

### Option 3: Plain Docker (No Compose)

```bash
# Build image
docker build -t sokegraph:latest .

# Run container
docker run -d \
  --name sokegraph-app \
  -p 8501:8501 \
  -v $(pwd)/data/outputs:/app/data/outputs \
  -v $(pwd)/external/output:/app/external/output \
  sokegraph:latest
```

---

## 📂 Directory Structure & Data Access

### How Docker Volumes Work

Docker uses **volume mounts** to share directories between your computer (host) and the container:

```
Your Computer                Docker Container
┌─────────────────────┐    ┌──────────────────────┐
│ ./data/outputs/     │◄───┤ /app/data/outputs/   │
│ ./external/output/  │◄───┤ /app/external/output/│
│ ./data/uploads/     │◄───┤ /app/data/uploads/   │
│ ./data/logs/        │◄───┤ /app/data/logs/      │
└─────────────────────┘    └──────────────────────┘
```

**This means:**
- ✅ Files created in the container appear on your computer immediately
- ✅ You can access all outputs without stopping the container
- ✅ Data persists even if you restart/rebuild the container
- ✅ You can provide input files by placing them in `./data/uploads/`

### What Gets Created Where

| Output Type        | Container Path          | Your Computer Path   | Description                 |
| ------------------ | ----------------------- | -------------------- | --------------------------- |
| Ranked papers CSVs | `/app/data/outputs/`    | `./data/outputs/`    | Paper rankings, scores      |
| Pipeline outputs   | `/app/external/output/` | `./external/output/` | GraphML, JSON, JSONLD files |
| Uploaded files     | `/app/data/uploads/`    | `./data/uploads/`    | PDFs, queries, ontologies   |
| Application logs   | `/app/data/logs/`       | `./data/logs/`       | Debug and error logs        |

### Example: Accessing Generated Files

```bash
# After running the pipeline, your files are here:
ls -la ./data/outputs/
# Papers_ranked_dynamic_high.csv
# Papers_ranked_static_high.csv
# updated_ontology.json

ls -la ./external/output/
# shared_ranked_by_pairs_then_mentions_dynamic_high.graphml
# shared_ranked_by_pairs_then_mentions_dynamic_high.json
```

---

## 🔧 Configuration

### Environment Variables

Edit `.env` file to configure:

```env
# Device selection
SOKEGRAPH_DEVICE=cpu  # Use 'cuda' for GPU, 'mps' for Mac M1/M2

# API Keys (if needed)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

### Port Configuration

Change ports in `docker-compose.yml`:

```yaml
ports:
  - "8080:8501"  # Access at localhost:8080 instead of 8501
```

---

## 📊 Managing the Container

### View Logs

```bash
# Follow logs in real-time
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# View only error logs
docker-compose logs | grep ERROR
```

### Stop/Start/Restart

```bash
# Stop the application
docker-compose down

# Start again
docker-compose up -d

# Restart (keeps volumes)
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build
```

### Access Container Shell

```bash
# Get a shell inside the container
docker exec -it sokegraph-streamlit bash

# Check Python environment
docker exec -it sokegraph-streamlit python --version

# Run a Python script
docker exec -it sokegraph-streamlit python -c "import sokegraph; print('OK')"
```

### Clean Up

```bash
# Stop and remove containers (keeps volumes)
docker-compose down

# Remove containers AND volumes (deletes data!)
docker-compose down -v

# Remove Docker image
docker rmi sokegraph:latest

# Clean everything including build cache
docker system prune -a
```

---

## 🚀 Production Deployment

### Using with Neo4j

Uncomment the Neo4j service in `docker-compose.yml` and start both:

```bash
docker-compose up -d
```

Access:
- **Streamlit App**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474 (user: neo4j, pass: password123)

### Resource Limits

The `docker-compose.yml` includes resource limits. Adjust as needed:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Max 4 CPU cores
      memory: 8G       # Max 8GB RAM
```

### Running on a Server

1. **Install Docker** on your server
2. **Clone the repository**
3. **Set environment variables** in `.env`
4. **Use a reverse proxy** (Nginx/Traefik) for HTTPS:

```nginx
# Nginx example
location / {
    proxy_pass http://localhost:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

---

## ❓ FAQ

### Q: Can I move the data directories elsewhere?

Yes! Edit the volume paths in `docker-compose.yml`:

```yaml
volumes:
  - /path/to/my/outputs:/app/data/outputs
  - /another/location:/app/external/output
```

### Q: How do I use GPU acceleration?

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Add to `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Set `SOKEGRAPH_DEVICE=cuda` in `.env`

### Q: Can I run multiple instances?

Yes, change the ports:

```bash
# Instance 1 (default)
docker-compose up -d

# Instance 2 (different port)
docker run -d -p 8502:8501 --name sokegraph-2 sokegraph:latest
```

### Q: How do I update the code?

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up -d --build
```

### Q: What if files aren't appearing on my computer?

Check volume mounts:

```bash
# Inspect container mounts
docker inspect sokegraph-streamlit | grep Mounts -A 20

# Verify permissions
ls -la data/
```

---

## 🛠 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs

# Common issues:
# - Port 8501 already in use: change port in docker-compose.yml
# - Permission errors: run `chmod -R 755 data/`
```

### Can't access files

```bash
# Check if volumes are mounted
docker exec sokegraph-streamlit ls -la /app/data/outputs

# Check host directory permissions
ls -la ./data/outputs
```

### Slow performance

```bash
# Increase resource limits in docker-compose.yml
# Or use GPU acceleration (see FAQ above)
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Docker Compose Reference](https://docs.docker.com/compose/)

---

## 🎉 You're Ready!

Your SOKEGraph application is now containerized and ready to run anywhere Docker is available!

**Next steps:**
1. Run `./docker-run.sh` to start
2. Access http://localhost:8501
3. Upload your files and run the pipeline
4. Check `./data/outputs/` for results
