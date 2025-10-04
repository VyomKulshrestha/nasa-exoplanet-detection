# 🚀 Deployment Guide for exoplanet.design

## Option 1: Streamlit Cloud (Recommended - FREE)

### Step 1: Prepare Your Repository

1. **Create a GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "NASA Exoplanet Detection System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/nasa-exoplanet-detection.git
   git push -u origin main
   ```

2. **Make sure these files are in your repo:**
   - ✅ `app.py` (main application)
   - ✅ `requirements.txt` (dependencies)
   - ✅ `.gitignore` (to exclude large files)
   - ✅ `README.md` (documentation)

### Step 2: Deploy to Streamlit Cloud

1. **Go to:** https://share.streamlit.io/
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Select:**
   - Repository: `YOUR_USERNAME/nasa-exoplanet-detection`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click** "Deploy"
6. **Wait 2-3 minutes** for deployment

You'll get a URL like: `https://YOUR_USERNAME-nasa-exoplanet-detection.streamlit.app`

### Step 3: Connect Your Custom Domain (exoplanet.design)

#### On Streamlit Cloud:
1. Go to your app settings
2. Click **"Custom domain"**
3. Enter: `exoplanet.design`
4. You'll get DNS instructions

#### On GoDaddy:
1. Log into GoDaddy
2. Go to **My Products** → **DNS** for exoplanet.design
3. Add these records:

**A Record:**
```
Type: A
Name: @
Value: [IP from Streamlit]
TTL: 600
```

**CNAME Record:**
```
Type: CNAME
Name: www
Value: [URL from Streamlit].streamlit.app
TTL: 600
```

4. **Save changes**
5. **Wait 10-30 minutes** for DNS propagation

🎉 Your site will be live at **https://exoplanet.design**!

---

## Option 2: DigitalOcean App Platform (Easy)

### Cost: ~$5/month

1. **Create DigitalOcean account:** https://www.digitalocean.com/
2. **Connect GitHub repository**
3. **Configure:**
   - Environment: Python
   - Run command: `streamlit run app.py --server.port=$PORT`
4. **Deploy**
5. **Add custom domain** in DigitalOcean settings
6. **Update GoDaddy DNS:**
   ```
   Type: A
   Name: @
   Value: [DigitalOcean IP]
   ```

---

## Option 3: AWS EC2 + Nginx (Full Control)

### Cost: ~$3.50/month (t3.micro)

### Step 1: Launch EC2 Instance
```bash
# Choose Ubuntu 22.04 LTS
# t3.micro instance (1GB RAM)
# Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)
```

### Step 2: Connect and Install
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv nginx -y

# Clone your repository
git clone https://github.com/YOUR_USERNAME/nasa-exoplanet-detection.git
cd nasa-exoplanet-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Create Systemd Service
```bash
sudo nano /etc/systemd/system/exoplanet.service
```

Add this:
```ini
[Unit]
Description=NASA Exoplanet Detection App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/nasa-exoplanet-detection
Environment="PATH=/home/ubuntu/nasa-exoplanet-detection/venv/bin"
ExecStart=/home/ubuntu/nasa-exoplanet-detection/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0

[Install]
WantedBy=multi-user.target
```

Start the service:
```bash
sudo systemctl start exoplanet
sudo systemctl enable exoplanet
```

### Step 4: Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/exoplanet.design
```

Add this:
```nginx
server {
    listen 80;
    server_name exoplanet.design www.exoplanet.design;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/exoplanet.design /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 5: Add SSL with Let's Encrypt
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d exoplanet.design -d www.exoplanet.design
```

### Step 6: Update GoDaddy DNS
```
Type: A
Name: @
Value: [Your EC2 IP]
TTL: 600

Type: A
Name: www
Value: [Your EC2 IP]
TTL: 600
```

---

## Option 4: Docker + Any Cloud Provider

### Create Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and deploy:
```bash
docker build -t exoplanet-detector .
docker run -p 8501:8501 exoplanet-detector
```

Deploy this container to:
- AWS ECS
- Google Cloud Run
- Azure Container Instances
- Railway.app
- Render.com

---

## 🎯 Recommendation

**For NASA Space Apps Challenge:**

1. **Start with Streamlit Cloud** (FREE, 2 minutes setup)
   - Perfect for demo and presentation
   - Automatic HTTPS
   - Easy updates via GitHub
   - Custom domain support

2. **Upgrade to AWS EC2 later** if you need:
   - Full control
   - Better performance
   - Custom configurations
   - Lower long-term costs

---

## 📊 Cost Comparison

| Platform | Monthly Cost | Setup Time | Custom Domain | SSL |
|----------|-------------|------------|---------------|-----|
| **Streamlit Cloud** | FREE | 2 min | ✅ Yes | ✅ Auto |
| **DigitalOcean** | $5 | 10 min | ✅ Yes | ✅ Auto |
| **AWS EC2** | $3.50 | 30 min | ✅ Yes | ✅ Manual |
| **Railway.app** | $5 | 5 min | ✅ Yes | ✅ Auto |
| **Render.com** | FREE/$7 | 5 min | ✅ Yes | ✅ Auto |

---

## 🚨 Before Deploying

Make sure your `.gitignore` includes:
```
__pycache__/
*.pyc
data/raw/*.pkl
data/processed/*.npy
.env
.streamlit/secrets.toml
```

---

## 🎉 After Deployment

1. **Test your site:** https://exoplanet.design
2. **Share with judges:** Send them the link
3. **Monitor uptime:** Use UptimeRobot (free)
4. **Analytics:** Add Google Analytics if needed

---

## 💡 Pro Tips

- **Use Streamlit Cloud for hackathon** (fastest)
- **DNS takes 10-30 min** to propagate
- **Test with www subdomain** too
- **Keep GitHub repo public** for easy judging
- **Add README with demo link** in your repo

---

## 📞 Need Help?

If you encounter issues:
1. Check DNS propagation: https://dnschecker.org/
2. Test without custom domain first
3. Check Streamlit Cloud logs
4. Verify GoDaddy DNS settings

Good luck with your deployment! 🚀
