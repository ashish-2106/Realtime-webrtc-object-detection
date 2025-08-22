FROM nginx:1.27-alpine
RUN rm -rf /usr/share/nginx/html/*
COPY public/ /usr/share/nginx/html/
COPY src/ /usr/share/nginx/html/src/
# pass envs to frontend at runtime via simple template
RUN printf 'env[MODE];env[SERVER_WS_URL];env[METRICS_POST_URL];env[METRICS_GET_URL]\n' > /usr/share/nginx/html/.env.hints
