functions=[
            {
                "name": "document_search",
                "description": "Search Documents for relvent information to user's query ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User query"},
                        "document_type": {"type": "string", "description": "Which document to search in", "enum": ["daggerheart_srd", "dungeon_master_srd"]},

                    },
                    "required": ["source"]
                }
            }

        ]