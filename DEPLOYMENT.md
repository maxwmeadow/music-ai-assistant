# Deployment Guide

This guide walks you through deploying the Music AI Assistant to production using Vercel (frontend) and Railway (backend + runner).

## Architecture

- **Frontend (Next.js)** → Vercel
- **Backend (FastAPI)** → Railway
- **Runner (Express)** → Railway

## Prerequisites

- GitHub repository with your code
- Vercel account (vercel.com)
- Railway account (railway.app)

---

## Step-by-Step Deployment

### Phase 1: Deploy Runner to Railway

**Why first?** The backend needs the runner URL to communicate with it.

1. **Create New Railway Project**
   - Go to railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `music-ai-assistant` repository
   - Name it: `music-ai-runner`

2. **Configure Service Settings**
   - Click on the service
   - Go to "Settings"
   - Set **Root Directory**: `runner`
   - Railway will auto-detect Node.js and use `npm start`

3. **Add Environment Variables**
   - Go to "Variables" tab
   - Add: `ALLOWED_ORIGINS` = `*` (temporary - we'll update this later)
   - Railway automatically sets `PORT`

4. **Deploy**
   - Railway will automatically deploy
   - Wait for deployment to complete
   - Go to "Settings" → "Networking" → "Generate Domain"
   - **Copy the domain** (e.g., `music-ai-runner-production.up.railway.app`)
   - **SAVE THIS URL** - you'll need it for the backend

5. **Test**
   - Visit: `https://your-runner-domain.railway.app/health`
   - Should return: `{"status":"ok","service":"music-runner","version":"2.0.0"}`

---

### Phase 2: Deploy Backend to Railway

1. **Create New Railway Project**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `music-ai-assistant` repository
   - Name it: `music-ai-backend`

2. **Configure Service Settings**
   - Click on the service
   - Go to "Settings"
   - Set **Root Directory**: `backend`
   - Railway will auto-detect the Dockerfile

3. **Add Environment Variables**
   - Go to "Variables" tab
   - Add these variables:

   ```
   ALLOWED_ORIGINS=*
   RUNNER_INGEST_URL=https://your-runner-domain.railway.app/eval
   RUNNER_INBOX_PATH=
   REQUEST_TIMEOUT_S=10
   ```

   Replace `your-runner-domain` with the URL from Phase 1

4. **Deploy**
   - Railway will build the Docker container
   - Wait for deployment (may take 3-5 minutes for first build)
   - Go to "Settings" → "Networking" → "Generate Domain"
   - **Copy the domain** (e.g., `music-ai-backend-production.up.railway.app`)
   - **SAVE THIS URL** - you'll need it for the frontend

5. **Test**
   - Visit: `https://your-backend-domain.railway.app/health`
   - Should return health check response

---

### Phase 3: Deploy Frontend to Vercel

1. **Import Project to Vercel**
   - Go to vercel.com
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Name it: `music-ai-assistant`

2. **Configure Project Settings**
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next` (default)
   - **Install Command**: `npm install`

3. **Add Environment Variables**
   - In "Environment Variables" section:

   ```
   NEXT_PUBLIC_API_URL=https://your-backend-domain.railway.app
   ```

   Replace `your-backend-domain` with the URL from Phase 2

4. **Deploy**
   - Click "Deploy"
   - Wait for build to complete (2-4 minutes)
   - **Copy your Vercel URL** (e.g., `music-ai-assistant.vercel.app`)

5. **Test**
   - Visit your Vercel URL
   - The app should load (API calls may fail until CORS is updated)

---

### Phase 4: Update CORS Settings

Now that all services are deployed, update CORS to allow cross-origin requests.

#### Update Runner CORS

1. Go to Railway → Runner project → Variables
2. Update `ALLOWED_ORIGINS`:
   ```
   https://your-frontend.vercel.app,https://your-backend-domain.railway.app
   ```
3. Railway will auto-redeploy

#### Update Backend CORS

1. Go to Railway → Backend project → Variables
2. Update `ALLOWED_ORIGINS`:
   ```
   https://your-frontend.vercel.app
   ```
3. Railway will auto-redeploy

---

## Verification Checklist

After all deployments:

- [ ] Runner health check works: `https://your-runner.railway.app/health`
- [ ] Backend health check works: `https://your-backend.railway.app/health`
- [ ] Frontend loads: `https://your-app.vercel.app`
- [ ] Can record and generate melody (tests backend → runner communication)
- [ ] Can compile and play music (tests frontend → backend → runner flow)

---

## Environment Variables Summary

### Runner (Railway)
```
ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://your-backend.railway.app
```

### Backend (Railway)
```
ALLOWED_ORIGINS=https://your-frontend.vercel.app
RUNNER_INGEST_URL=https://your-runner.railway.app/eval
RUNNER_INBOX_PATH=
REQUEST_TIMEOUT_S=10
```

### Frontend (Vercel)
```
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

---

## Troubleshooting

### Frontend can't connect to backend
- Check `NEXT_PUBLIC_API_URL` in Vercel environment variables
- Check backend CORS settings include your Vercel domain
- Check browser console for errors

### Backend can't connect to runner
- Check `RUNNER_INGEST_URL` in backend Railway variables
- Check runner CORS settings include your backend domain
- Check Railway logs for connection errors

### Build failures
- **Frontend**: Check Next.js build logs in Vercel
- **Backend**: Check Docker build logs in Railway
- **Runner**: Check npm install logs in Railway

### CORS errors
- Make sure all URLs in CORS settings use `https://` (not `http://`)
- No trailing slashes in URLs
- Comma-separated, no spaces

---

## Updating Deployments

### Frontend (Vercel)
- Push to main branch → Auto-deploys
- Or use Vercel dashboard → Deployments → Redeploy

### Backend & Runner (Railway)
- Push to main branch → Auto-deploys
- Or use Railway dashboard → Deployments → Redeploy

---

## Local Development After Deployment

Your local `.env.local` files still point to `localhost`, so local development works unchanged:

```bash
# Local development (unchanged)
npm run dev         # Frontend on localhost:3000
uvicorn ...         # Backend on localhost:8000
node server.js      # Runner on localhost:5001
```

---

## Cost Estimates

- **Vercel**: Free tier (Hobby plan) - sufficient for most use cases
- **Railway**: ~$5-10/month per service with usage-based pricing
  - Runner: ~$5/month (minimal compute)
  - Backend: ~$10/month (model inference compute)

---

## Support

If you encounter issues:
1. Check Railway logs: Dashboard → Service → Deployments → View Logs
2. Check Vercel logs: Dashboard → Project → Deployments → View Function Logs
3. Check browser console for frontend errors
4. Test each service's `/health` endpoint