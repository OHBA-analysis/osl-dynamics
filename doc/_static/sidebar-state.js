// Keep the left sidebar collapsed on every page once the reader collapses it,
// until they expand it again. The theme's "Collapse Sidebar" button doesn't
// remember its state, so it's saved in localStorage.
(function () {
  const KEY = "osl-dynamics-sidebar-collapsed";

  function load() {
    try {
      return localStorage.getItem(KEY) === "true";
    } catch (e) {
      return false;
    }
  }

  function save(collapsed) {
    try {
      localStorage.setItem(KEY, collapsed ? "true" : "false");
    } catch (e) {}
  }

  // Before the page is drawn: show the sidebar collapsed, without animating
  // (see custom.css)
  const collapsed = load();
  if (collapsed) {
    document.documentElement.classList.add("osld-sidebar-collapsed");
  }

  // Save the new state when the button is clicked. This listens on the
  // document in the capture phase so it sees the click before the theme's
  // handler, which changes the button's state
  document.addEventListener(
    "click",
    function (event) {
      const button = event.target.closest("#pst-collapse-sidebar-button");
      if (!button || button.dataset.busy === "true") return; // the theme ignores it too
      save(button.getAttribute("aria-expanded") === "true");
    },
    true
  );

  document.addEventListener("DOMContentLoaded", function () {
    const button = document.getElementById("pst-collapse-sidebar-button");
    const sidebar = document.getElementById("pst-primary-sidebar");
    if (collapsed && button && sidebar) {
      // The theme's own collapsed state
      sidebar.classList.add("pst-squeeze");
      button.setAttribute("aria-expanded", "false");
    }
    // Turn the animations back on once the page is drawn
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        document.documentElement.classList.remove("osld-sidebar-collapsed");
      });
    });
  });
})();
