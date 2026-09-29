# syntax=docker/dockerfile:1

# ---- build the bundle -------------------------------------------------------
FROM node:22-slim AS build

WORKDIR /app

# Lockfile first, so a source change does not reinstall dependencies.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ---- serve it ---------------------------------------------------------------
# nginx serves the static bundle and reverse-proxies /api to the backend, so the
# browser only ever talks to one origin - no CORS, no build-time API URL.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:3000/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
