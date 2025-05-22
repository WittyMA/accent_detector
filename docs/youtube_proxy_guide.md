# YouTube Proxy Service Guide

This guide explains how to use the proxy service solution to bypass YouTube's authentication requirements.

## Understanding the Issue

When using the Accent Detection System with YouTube videos, you might encounter this error:

```
ERROR: [youtube] Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies for the authentication.
```

This happens because YouTube's anti-bot systems are triggered when:
- Making multiple requests in a short time
- Accessing from cloud/server environments (like Streamlit Cloud)
- Accessing region-restricted or age-restricted content

## Solution: Using Proxy Services

The `video_processor_proxy.py` implementation uses multiple proxy services to bypass YouTube's authentication requirements. This approach has several advantages:

- **No cookies required**: Works without browser cookies or authentication
- **Multiple fallback methods**: Tries several different proxy services if one fails
- **Cloud-friendly**: Ideal for Streamlit Cloud and other server environments
- **Simpler deployment**: No need to manage cookie files

## How to Use the Proxy Solution

1. Replace your current `video_processor.py` with the provided `video_processor_proxy.py`:
   ```bash
   # Rename the new file to replace the old one
   mv src/utils/video_processor_proxy.py src/utils/video_processor.py
   ```

2. No additional configuration is needed - the proxy solution works out of the box

## How It Works

The proxy solution uses several methods to bypass YouTube restrictions:

1. **Invidious Instances**: Public front-ends for YouTube that don't require authentication
2. **Alternative Domains**: YouTube mirrors and alternative access points
3. **Public APIs**: Services like Piped that provide direct access to YouTube content
4. **Multiple Fallbacks**: If one method fails, it automatically tries others

## Limitations

While the proxy solution is more convenient, it does have some limitations:

1. **Reliability**: Proxy services may occasionally be unavailable or blocked
2. **Video Availability**: Some highly restricted videos might still be inaccessible
3. **Performance**: May be slightly slower than direct access with authentication
4. **Future Changes**: YouTube or proxy services may change their APIs

## Troubleshooting

If you encounter issues:

1. **Try different videos**: Some videos have stricter restrictions than others
2. **Check proxy status**: The public proxy services might be temporarily down
3. **Update the code**: You might need to update the proxy service URLs if they change
4. **Fallback to cookie method**: For highly restricted videos, the browser cookie method might be necessary

## Alternative Solutions

If the proxy service approach doesn't work for your needs:

1. **Browser Cookie Method**: See the `youtube_cookie_guide.md` for instructions on using browser cookies
2. **Host Your Own Videos**: For testing, consider hosting sample videos on your own server
3. **Use Public Domain Videos**: Sources like Archive.org offer unrestricted video content
