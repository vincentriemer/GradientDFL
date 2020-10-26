FROM ufoym/deepo:all-py36-cu100

COPY scripts /scripts

ENTRYPOINT [ "bash", "/scripts/run.sh" ]
