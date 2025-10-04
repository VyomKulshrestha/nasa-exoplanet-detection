@echo off
echo ========================================
echo NASA Exoplanet Detection - GitHub Setup
echo ========================================
echo.

echo Step 1: Initializing Git repository...
git init
echo.

echo Step 2: Adding all files...
git add .
echo.

echo Step 3: Creating first commit...
git commit -m "Initial commit: NASA Exoplanet Detection System for Space Apps Challenge"
echo.

echo Step 4: Setting main branch...
git branch -M main
echo.

echo ========================================
echo NEXT STEPS:
echo ========================================
echo.
echo 1. Create a GitHub repository at: https://github.com/new
echo    Name it: nasa-exoplanet-detection
echo    Make it PUBLIC (required for Streamlit Cloud free tier)
echo.
echo 2. Copy the repository URL (it will look like):
echo    https://github.com/YOUR_USERNAME/nasa-exoplanet-detection.git
echo.
echo 3. Run these commands (replace YOUR_USERNAME):
echo    git remote add origin https://github.com/YOUR_USERNAME/nasa-exoplanet-detection.git
echo    git push -u origin main
echo.
echo 4. Go to: https://share.streamlit.io/
echo    - Sign in with GitHub
echo    - Click "New app"
echo    - Select your repository
echo    - Set main file: app.py
echo    - Deploy!
echo.
echo 5. Connect your domain exoplanet.design in Streamlit settings
echo.
echo ========================================
echo Ready to deploy! Good luck! 🚀
echo ========================================
pause

