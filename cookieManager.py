# import streamlit as st
# from streamlit_js_eval import streamlit_js_eval
# import json
# from typing import Optional, Any

# class CookieManager:
#     @staticmethod
#     def set_cookie(name: str, value: Any, days: int = 30) -> None:
#         """
#         Set a cookie with proper encoding and expiration
        
#         Args:
#             name: Cookie name
#             value: Cookie value (will be JSON serialized)
#             days: Number of days until cookie expires
#         """
#         try:
#             # JSON serialize the value to handle various data types
#             json_value = json.dumps(value)
#             # Calculate expiration
#             expiry = f"max-age={days * 24 * 60 * 60}"
#             # Set cookie with proper attributes
#             js_code = f"""
#                 document.cookie = '{name}={json_value}; path=/; {expiry}; SameSite=Lax';
#                 true;  // Return value to ensure execution
#             """
#             streamlit_js_eval(js_expressions=js_code)
#             # Store in session state
#             if 'cookies' not in st.session_state:
#                 st.session_state.cookies = {}
#             st.session_state.cookies[name] = value
            
#         except Exception as e:
#             st.error(f"Error setting cookie: {str(e)}")
    
#     @staticmethod
#     def get_cookie(name: str, default: Any = None) -> Optional[Any]:
#         """Get a cookie value with fallback to session state"""
#         try:
#             # First try session state to avoid JS calls
#             if 'cookies' in st.session_state and name in st.session_state.cookies:
#                 return st.session_state.cookies[name]
            
#             # If not in session state, try browser cookies once
#             if not hasattr(st.session_state, '_cookies_loaded'):
#                 js_code = """
#                     (function() {
#                         const cookies = document.cookie.split(';').reduce((acc, curr) => {
#                             const [key, value] = curr.trim().split('=');
#                             try {
#                                 acc[key] = JSON.parse(decodeURIComponent(value));
#                             } catch {
#                                 acc[key] = decodeURIComponent(value);
#                             }
#                             return acc;
#                         }, {});
#                         return JSON.stringify(cookies);
#                     })();
#                 """
#                 try:
#                     cookies_str = streamlit_js_eval(js_expressions=js_code, key='load_all_cookies')
#                     if cookies_str:
#                         cookies = json.loads(cookies_str)
#                         if 'cookies' not in st.session_state:
#                             st.session_state.cookies = {}
#                         st.session_state.cookies.update(cookies)
#                 except:
#                     pass
#                 st.session_state._cookies_loaded = True
            
#             # Return from session state or default
#             return st.session_state.cookies.get(name, default)

#         except Exception as e:
#             st.error(f"Error getting cookie: {str(e)}")
#             return default