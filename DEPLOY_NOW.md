# 🚀 Deploy to exoplanet.design RIGHT NOW!

## ⚡ Quick Start (5 Minutes)

### Step 1: Push to GitHub (2 minutes)

1. **Create a GitHub repository:**
   - Go to: https://github.com/new
   - Name: `nasa-exoplanet-detection`
   - **Make it PUBLIC** ✅
   - Click "Create repository"

2. **Push your code:**
   ```bash
   # Run the batch file (Windows)
   deploy_to_github.bat
   
   # Then run these (replace YOUR_USERNAME):
   git remote add origin https://github.com/YOUR_USERNAME/nasa-exoplanet-detection.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud (2 minutes)

1. **Go to:** https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click:** "New app"
4. **Fill in:**
   - Repository: `YOUR_USERNAME/nasa-exoplanet-detection`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click:** "Deploy!"
6. **Wait 2-3 minutes** ☕

You'll get a URL like: `https://YOUR-APP.streamlit.app`

### Step 3: Connect Your Domain (1 minute)

#### In Streamlit Cloud:
1. Click **Settings** (⚙️) on your deployed app
2. Click **"Custom domain"**
3. Enter: `exoplanet.design`
4. Copy the DNS records shown

#### In GoDaddy:
1. Go to: https://dcc.godaddy.com/
2. Click your domain **exoplanet.design**
3. Click **DNS** → **Manage DNS**
4. **Delete** existing A/CNAME records for @ and www
5. **Add new records** from Streamlit

**Typical setup:**
```
Type: CNAME
Name: @
Value: YOUR-APP.streamlit.app
TTL: 600 (or 1 hour)

Type: CNAME  
Name: www
Value: YOUR-APP.streamlit.app
TTL: 600
```

6. **Save** and wait 10-30 minutes for DNS to propagate

---

## ✅ Verification

1. **Check deployment:**
   - Your app should be live at the Streamlit URL immediately
   
2. **Check domain (after 10-30 min):**
   - Visit: https://exoplanet.design
   - Visit: https://www.exoplanet.design
   
3. **Test DNS propagation:**
   - https://dnschecker.org/ (enter exoplanet.design)

---

## 🎯 You're Live!

Once DNS propagates, your NASA Exoplanet Detection System will be live at:

**🌐 https://exoplanet.design**

Share this with:
- NASA Space Apps Challenge judges
- Social media
- Your team
- The world! 🌍

---

## 🔄 Updating Your Site

To update your site after deployment:

```bash
# Make changes to your code
# Then:
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud will automatically redeploy!
```

---

## 🆘 Troubleshooting

### "Module not found" errors:
- Make sure `requirements.txt` is in your repo
- Check Streamlit Cloud logs

### Domain not working:
- Wait 30 minutes for DNS
- Check GoDaddy DNS settings
- Use https:// (not http://)
- Try www.exoplanet.design

### App is slow:
- First load downloads NASA data (takes 1-2 min)
- Subsequent loads are cached and fast

---

## 💡 Pro Tips

1. **Use a custom README.md** with screenshots
2. **Add a favicon** in `.streamlit/`
3. **Share the direct link** before DNS propagates
4. **Monitor with Streamlit Analytics** (built-in)
5. **Keep repo public** for free hosting

---

## 📊 What You Get

✅ **FREE hosting** on Streamlit Cloud  
✅ **Custom domain** (exoplanet.design)  
✅ **Automatic HTTPS** (secure)  
✅ **Auto-deploy** on git push  
✅ **Built-in analytics**  
✅ **Always-on** (never sleeps)  

---

## 🎉 Ready to Launch?

Run the batch file and follow the steps above!

```bash
deploy_to_github.bat
```

**Good luck with the NASA Space Apps Challenge! 🚀🌟**
