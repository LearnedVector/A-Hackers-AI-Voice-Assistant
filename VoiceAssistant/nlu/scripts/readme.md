## NLU dataset example ##
![Alt text](./images/dataset-example.png?raw=true "NLU dataset")


## Intent and Scenario Dataframe Stats
**18 Scenarios total**
|Scenario| Count|
|--------|--------------|
|general|          6102|
|calendar|          2986|
|play|              2613|
|qa|                2370|
|email|             1764|
|iot|               1327|
|weather|           1062|
|transport|         1023|
|lists|              993|
|news|               877|
|social|             727|
|datetime|           723|
|recommendation|     693|
|alarm|              623|
|music|              585|
|cooking|            421|
|takeaway|           414|
|audio|              412|

**54 Intent total**
|Intent|             Count|
|--------|--------------|
|query|              5980|
|set|                1748|
|music|              1205|
|quirky|             1088|
|factoid|            1052|
|remove|              986|
|negate|              939|
|praise|              785|
|sendemail|           694|
|explain|             684|
|repeat|              585|
|affirm|              554|
|radio|               551|
|confirm|             550|
|post|                541|
|definition|          504|
|dontcare|            450|
|recipe|              415|
|podcasts|            379|
|currency|            378|
|events|              324|
|commandstop|         320|
|createoradd|         294|
|stock|               270|
|locations|           257|
|hue_lightoff|        246|
|audiobook|           241|
|ticket|              239|
|game|                237|
|hue_lightchange|     224|
|querycontact|        221|
|likeness|            204|
|traffic|             200|
|order|               199|
|coffee|              198|
|taxi|                185|
|cleaning|            172|
|maths|               166|
|volume_mute|         163|
|volume_up|           145|
|hue_lightup|         142|
|hue_lightdim|        126|
|joke|                122|
|movies|              112|
|wemo_off|            100|
|convert|              97|
|addcontact|           90|
|settings|             80|
|wemo_on|              80|
|volume_down|          80|
|hue_lighton|          39|
|dislikeness|          25|
|greet|                25|
|volume_other|         24|
**Stats**

| Column         | Non-Null Count    | Dtype| 
|----------      |--------------     |----- 
| 0   Sentence # |  25715 non-null  |int64 
| 1   scenario    | 25715 non-null  |object
| 2   intent      | 25715 non-null  |object


## Entity Dataframe Distribution and Stats
    - The O token represents the out-of-entity token for non-entity words. 
    - Total entites(O inclisive) is 57
    - Clearly skew distribution
|Entity|                   Count|                  
|----------      |-------------- | 
|O|                       140605|
|date|                      4652|
|place_name|                3047|
|time|                      2974|
|event_name|                2696|
|person|                    2200|
|business_name|             1038|
|media_type|                1027|
|radio_name|                 955|
|currency_name|              901|
|device_type|                804|
|weather_descriptor|         752|
|food_type|                  744|
|artist_name|                662|
|news_topic|                 660|
|song_name|                  601|
|list_name|                  567|
|definition_word|            557|
|transport_type|             540|
|relation|                   479|
|timeofday|                  470|
|house_place|                435|
|music_genre|                407|
|business_type|              369|
|player_setting|             350|
|game_name|                  331|
|audiobook_name|             298|
|podcast_descriptor|         275|
|general_frequency|          267|
|email_address|              239|
|playlist_name|              214|
|personal_info|              212|
|order_type|                 207|
|podcast_name|               207|
|color_type|                 178|
|change_amount|              166|
|music_descriptor|           145|
|time_zone|                  132|
|meal_type|                  113|
|app_name|                    76|
|joke_type|                   73|
|transport_agency|            72|
|email_folder|                55|
|transport_name|              54|
|movie_name|                  54|
|ingredient|                  52|
|coffee_type|                 50|
|transport_descriptor|        33|
|audiobook_author|            31|
|alarm_type|                  30|
|cooking_type|                27|
|movie_type|                  24|
|drink_type|                  23|
|sport_type|                  17|
|music_album|                  7|
|query_detail|                 6|
|game_type|                    3|

|   Column |      Non-Null Count|   Dtype | 
|---------|  --------------|   -----    | 
|  Sentence # | 172163 non-null |  int64 |
|  words |       172163 non-null |  object|
|  entity|      172163 non-null|  object|



### Dataset references ###


@InProceedings{XLiu.etal:IWSDS2019,
  author    = {Xingkun Liu, Arash Eshghi, Pawel Swietojanski and Verena Rieser},
  title     = {Benchmarking Natural Language Understanding Services for building Conversational Agents},
  booktitle = {Proceedings of the Tenth International Workshop on Spoken Dialogue Systems Technology (IWSDS)},
  month     = {April},
  year      = {2019},
  address   = {Ortigia, Siracusa (SR), Italy},
  publisher = {Springer},
  pages     = {xxx--xxx},
  url       = {http://www.xx.xx/xx/}
}
https://github.com/xliuhw/NLU-Evaluation-Data