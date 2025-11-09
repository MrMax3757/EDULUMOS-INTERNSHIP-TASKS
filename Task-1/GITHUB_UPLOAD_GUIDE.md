# How to Upload Project to GitHub

## Step-by-Step Guide

### Option 1: If you already have a GitHub repository

1. **Navigate to your project directory** (if not already there):
   ```bash
   cd D:\EduLumos-Internship-Tasks\Task-1
   ```

2. **Check current git status**:
   ```bash
   git status
   ```

3. **Add all files to staging**:
   ```bash
   git add .
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "Initial commit: Add Smart Study Score Predictor project"
   ```

5. **Push to GitHub**:
   ```bash
   git push origin main
   ```
   (Use `git push origin master` if your default branch is `master`)

### Option 2: Create a new GitHub repository

#### On GitHub Website:

1. **Go to GitHub.com** and sign in
2. **Click the "+" icon** in the top right corner
3. **Select "New repository"**
4. **Repository name**: `Smart-Study-Score-Predictor` (or any name you prefer)
5. **Description**: "Machine learning project to predict student performance scores"
6. **Visibility**: Choose Public or Private
7. **DO NOT** initialize with README, .gitignore, or license (since you already have files)
8. **Click "Create repository"**

#### In Your Terminal:

1. **Navigate to your project directory**:
   ```bash
   cd D:\EduLumos-Internship-Tasks\Task-1
   ```

2. **Initialize git (if not already done)**:
   ```bash
   git init
   ```

3. **Add remote repository** (replace `YOUR_USERNAME` with your GitHub username):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Smart-Study-Score-Predictor.git
   ```
   Or if using SSH:
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/Smart-Study-Score-Predictor.git
   ```

4. **Add all files**:
   ```bash
   git add .
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Initial commit: Add Smart Study Score Predictor project"
   ```

6. **Rename branch to main** (if needed):
   ```bash
   git branch -M main
   ```

7. **Push to GitHub**:
   ```bash
   git push -u origin main
   ```

### Option 3: Using GitHub CLI (if installed)

1. **Navigate to your project directory**:
   ```bash
   cd D:\EduLumos-Internship-Tasks\Task-1
   ```

2. **Create and push repository**:
   ```bash
   gh repo create Smart-Study-Score-Predictor --public --source=. --remote=origin --push
   ```

## Troubleshooting

### If you get "fatal: remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### If you get authentication errors:
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys for authentication

### If you want to update existing repository:
```bash
git add .
git commit -m "Update: Add README and project files"
git push origin main
```

## Quick Command Summary

```bash
# Navigate to project
cd D:\EduLumos-Internship-Tasks\Task-1

# Check status
git status

# Add all files
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

## Notes

- The `.gitignore` file will exclude unnecessary files like `__pycache__`, `.ipynb_checkpoints`, etc.
- Large files (like the CSV and model files) will be uploaded. If they're too large, consider using Git LFS or storing them elsewhere.
- Make sure you have the latest version of git installed.
- If you encounter any issues, check your git configuration:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```

