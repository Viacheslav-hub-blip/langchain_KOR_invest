from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain_community.llms import GigaChat

token = 'ZTk3ZjdmYjMtNmMwOC00NGE1LTk0MzktYzA3ZjU4Yzc2YWI3OmY2OGFlMTQ1LTIyNzgtNDIxMC05M2JmLWFhNTFkZjdmYTY1Yw=='
model = GigaChat(credentials=token, verify_ssl_certs=False)

schema = Object(
    id="invest_info",
    description="Personal information about invest wish",
    attributes=[
        Text(
            id="invest_object",
            description="the name of the investment object",
            examples=[
                ("gold and stocks are a good asset to invest in", "gold"),
                ("People are increasingly buying stocks", "stocks"),
                ("people used to buy shares of Tesla", "share Tesla"),
                ("John is thinking of buying Microsoft shares", "Microsoft share")
            ],
        ),
        Text(
            id="wish",
            description="human desire",
            examples=[("People often want to buy gold", "buy"),
                      ("people now want to sell Tesla shares", "sell"),
                      ("people hold Apple shares in their assets", "hold"),
                      ],
        ),

    ],
    examples=[
        (
            "Slava is thinking about buying gold",
            [
                {"invest_object": "gold", "wish": "buying"},
            ],
        ),
        (
            "John is thinking of selling Tesla shares",
            [
                {"invest_object": "Tesla share", "wish": "selling"}
            ]
        ),
        (
            "Smit is thinking of sell Oracle shares",
            [
                {"invest_object": "Oracle share", "wish": "sell"}
            ]
        ),
    ],
    many=True,
)

chain = create_extraction_chain(model, schema)
print(chain.run("I want to keep the gold and shares Tesla for another 2 years")["data"])
