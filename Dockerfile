FROM ufoym/deepo:all-py36

COPY scripts /scripts

ENTRYPOINT [ "bash", "/scripts/run.sh" ]
