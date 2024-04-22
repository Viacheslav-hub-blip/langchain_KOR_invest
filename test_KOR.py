from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain_community.llms import GigaChat

token = 'ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmY2OGFlMTQ1LTIyNzgtNDIxMC05M2JmLWFhNTFkZjdmYTY1Yw=='
model = GigaChat(credentials=token, verify_ssl_certs=False)

schema = Object(
    id="resource",
    description="resource information",
    examples=[
        ("gold and stocks are a good asset to invest in", [{"resource": "gold"}, {"resource": "stocks"}]),
        ("People are increasingly buying stocks", [{"resource": "stocks"}]),
        ("people used to buy shares of Tesla", [{"resource": "shares"}])
    ],
    attributes=[
        Text(
            id="resource",
            description="the name of the resource",
        )
    ],
    many=True,
)

chain = create_extraction_chain(model, schema)

print(chain.run(("стоит ли покупать золото в данный момент?"))["data"])
