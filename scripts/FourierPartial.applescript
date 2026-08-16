-- FourierPartial.app
--
-- An AppleScript applet, deliberately, rather than a shell script in a bundle.
--
-- WHY: macOS will not run a bundle's executable a second time while that
-- bundle is already running. It sends the running instance a "reopen" event
-- instead. A shell script cannot receive that event, so the second click did
-- nothing whatsoever -- no window, no error, no clue that anything had
-- happened. An applet receives it, which is the entire reason this file exists.
--
-- All the real work is in scripts/fp_launch.sh. This is only the event plumbing:
--
--   run     first launch          -> start the server, open a browser
--   reopen  clicked while running -> open a browser (server already up)
--   quit    user quits the app    -> stop the server
--
-- `ensure` is idempotent, so run and reopen do the same thing and clicking the
-- icon always ends with a window on a working app.

on projectRoot()
	-- The bundle lives inside the project, so its parent directory is the
	-- project root. Resolved at runtime rather than baked in, so the folder can
	-- be moved or renamed without breaking the app.
	set myPath to POSIX path of (path to me)
	return do shell script "cd " & quoted form of myPath & "/.. && pwd -P"
end projectRoot

on runLauncher(action)
	set root to projectRoot()
	set launcher to quoted form of (root & "/scripts/fp_launch.sh")
	-- `|| true` so a failure comes back as a token we can read rather than as
	-- an AppleScript error with the output buried in its message.
	return do shell script launcher & " " & action & " 2>&1 || true"
end runLauncher

on reportProblem(token)
	set root to projectRoot()
	if token contains "MISSING_VENV" then
		set msg to "The Python environment has not been set up yet." & return & return & ¬
			"Open Terminal and run:" & return & return & ¬
			"    cd " & root & return & "    ./setup.sh" & return & return & ¬
			"Then click FourierPartial again."
	else if token contains "NO_WRITE" then
		set msg to "Could not create the logs folder inside:" & return & return & ¬
			"    " & root & return & return & ¬
			"Check the folder is writable and not on a read-only volume."
	else
		set msg to "The server did not start." & return & return & ¬
			"The last lines of the log:" & return & return & ¬
			(do shell script "tail -n 12 " & quoted form of (root & "/logs/server.log") & " 2>/dev/null || echo '(no log was written)'") & ¬
			return & return & "Full log:" & return & "    " & root & "/logs/server.log"
	end if
	display dialog msg with title "fourier_partial" buttons {"OK"} default button "OK" with icon stop
end reportProblem

on openApp()
	set outcome to runLauncher("ensure")
	if outcome starts with "http" then
		-- Open the browser on the URL the server actually chose. It picks a
		-- free port when 8420 is taken, so assuming the port would sometimes
		-- open a window onto somebody else's server.
		do shell script "/usr/bin/open " & quoted form of outcome
	else
		reportProblem(outcome)
	end if
end openApp

on run
	openApp()
end run

on reopen
	-- THE FIX. Clicking the icon while the app is already running lands here.
	openApp()
end reopen

on quit
	try
		runLauncher("stop")
	end try
	continue quit
end quit
