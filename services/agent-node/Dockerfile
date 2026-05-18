FROM node:22-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-bookworm-slim
WORKDIR /app
# OpenCode SDK spawns the `opencode` CLI as its embedded server.
RUN npm install -g opencode-ai@1.15.4
COPY --from=deps /app/node_modules ./node_modules
COPY package.json package-lock.json tsconfig.json ./
COPY src ./src
RUN npm run build
ENV NODE_ENV=production
ENV PORT=13100
EXPOSE 13100
CMD ["node", "dist/server.js"]
