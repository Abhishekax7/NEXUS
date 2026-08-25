from dataclasses import dataclass

from ddgs import DDGS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchTool:
    def __init__(
        self,
        max_results: int = 5,
    ):
        self.max_results = max_results

    def search(
        self,
        query: str,
    ) -> list[SearchResult]:
        results = DDGS().text(
            query,
            max_results=self.max_results,
        )

        return [
            SearchResult(
                title=item.get(
                    "title",
                    "",
                ),
                url=item.get(
                    "href",
                    "",
                ),
                snippet=item.get(
                    "body",
                    "",
                ),
            )
            for item in results
        ]
