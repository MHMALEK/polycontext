FROM node:22-alpine

WORKDIR /app

# Copy manifest first so layer caching works for the dep install.
COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY server.js ./

ENV PORT=3040
ENV HOST=0.0.0.0
EXPOSE 3040

# Health: fastify root path doesn't exist; /health is the contract.
# Use Node's own fetch (built-in v22+) instead of wget so we don't depend on
# alpine's busybox utils, which can drop wget in slim variants.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://localhost:3040/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
