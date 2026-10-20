// The website's early theme check, from its base.html: turn dark mode on before
// the page paints, and a dark page doesn't flash light. Runs from <body>'s first
// line, where document.body exists.
(function () {
  try {
    if (localStorage.getItem('darkMode') === 'true') {
      document.body.classList.add('dark-mode');
      document.documentElement.classList.add('dark');
    }
  } catch (e) {
    // Storage can be blocked; the light theme is used then.
  }
})();
