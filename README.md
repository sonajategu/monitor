# Political Promise Tracker by Sõna ja Tegu
### _Tegude radar_ – 'Actions radar' in Estonian
A Python-based solution that monitors news articles for political promises and automatically tracks them in a database of Sõna ja Tegu platform ([sonajategu.ee](https://sonajategu.ee)).

## Description

This application monitors Estonian news sources (ERR, Delfi, Postimees ans other agencies) for political promises and commitments made by politicians and institutions. It automatically detects, extracts, and catalogues these promises using natural language processing (via OpenAI's GPT) and stores them in a structured format on "Sõna ja Tegu" platform.

All promises tracked by this application could be found on Sõna ja tegu webpage: https://sonajategu.ee/radar

## Features

- Automated monitoring of RSS feeds from Estonian news sources
- Intelligent promise detection using OpenAI's GPT API
- Parallel processing of news articles
- Redis-based caching to avoid duplicate processing
- Structured logging system for monitoring and debugging
- Integration for promise storage and management
- Rate limiting and error handling
- Support for multiple political parties and government institutions

## Roadmap
TBD

## Prerequisites

- Python 3.8+
- Redis server
- OpenAI API key
- JWT authentication token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sonajategu/monitor
cd monitor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following configuration:
```
WP_API_BASE=your-wordpress-site-url
WP_JWT_TOKEN=your-wordpress-jwt-token
OPENAI_API_KEY=your-openai-api-key
```

## Configuration

### News Sources

News sources are configured in `news_sources.py`. Each source is defined with:
- URL (RSS feed)
- Category
- Subcategory
- Name
- Priority
- Rate limit

### Logging

The application uses a structured logging system with separate logs for:
- Activity logging
- Error logging
- Performance logging
- OpenAI interactions

Logs are stored in the `logs` directory.

## Usage

Run the main application:
```bash
python monitor/tracker.py
```

The application will:
1. Monitor configured RSS feeds
2. Download and process articles
3. Detect promises using GPT
4. Send promises on Sõna ja Tegu
5. Cache processed articles
6. Run continuously with a 1-hour interval

## Promise Detection

Promises are detected and structured with the following information:
- Title
- Content
- Source URL
- Person making the promise
- Date
- Authority
- Topic
- Budget (if mentioned)
- Deadline (if mentioned)

## WordPress Integration

Promises are stored as custom post types in WordPress with:
- ACF fields for structured data
- Custom taxonomies for people, topics, and authorities
- Status tracking
- Source linking

## Error Handling

The application includes comprehensive error handling and logging:
- API call failures
- parsing errors
- network issues
- Rate limiting violations

## License

This project is protected under GNU AGPL license.

## Credits

"Tegude radar" project was developed with the support of our partners from [Baltic Centre for Media Excellence](https://www.bcme.eu/), [TechSoup](https://www.techsoup.org/),  [HiveMind Community](https://en.hive-mind.community/) and [Google.org](https://www.google.org/).

Read more about the Sõna ja tegu project here: https://sonajategu.ee/projektist/

If you would like to contribute or have a question, please contact our team by email at [hello@sonajategu.ee](mailto:hello@sonajategu.ee).
