import
function __fish_using_command
    set cmd (commandline -opc)
    for i in (seq (count $cmd) 1)
        switch $cmd[$i]
        case "-*"
        case "*"
            if [ $cmd[$i] = $argv[1] ]
                return 0
            else
                return 1
            end
        end
    end
    return 1
end

function __option_entered_check
    set cmd (commandline -opc)
    for i in (seq (count $cmd))
        switch $cmd[$i]
        case "-*"
            if [ $cmd[$i] = $argv[1] ]
                return 1
            end
        end
    end
    return 0
end

function __is_prev_global
    set cmd (commandline -opc)
    set global_options "--dollar-limit" "--file-loader-parallel-backend" "--path" "--exclude" "--anki-template" "--llm-verbosity" "--online-media-url-regex" "--summary-n-recursion" "--query-retrievers" "--llms-api-bases" "--embed-model" "--silent" "--save-embeds-as" "--notification-callback" "--DIY-rolling-window-embedding" "--private" "--filter-metadata" "--debug" "--whisper-prompt" "--whisper-lang" "--task" "--loading-failure" "--pdf-parsers" "--json-dict-template" "--anki-notetype" "--query-eval-check-number" "--file-loader-n-jobs" "--modelname" "--anki-profile" "--summary-language" "--filter-content" "--doccheck-min-lang-prob" "--query-eval-modelname" "--doccheck-max-token" "--embed-kwargs" "--youtube-audio-backend" "--disable-md-printing" "--anki-deck" "--online-media-resourcetype-regex" "--deepgram-kwargs" "--disable-llm-cache" "--query" "--audio-backend" "--audio-unsilence" "--import-mode" "--youtube-language" "--out-file" "--verbose" "--source-tag" "--include" "--json-dict-exclude-keys" "--filetype" "--top-k" "--query-relevancy" "--embed-instruct" "--load-embeds-from" "--youtube-translation" "--doccheck-min-token" "--version" "--anki-tag-filter" "--load-functions" "--anki-tag-render-filter"
    set prev (count $cmd)

    for opt in $global_options
        if [ "--$opt" = $cmd[$prev] ]
            echo $prev
            return 0
        end
    end
    return 1
end

complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --prefix' -l prefix
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command removeprefix; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command wdoc' -f -a allowed-extra-args
complete -c wdoc -n '__fish_using_command wdoc' -f -a prepare-query-task
complete -c wdoc -n '__fish_using_command wdoc' -f -a VERSION
complete -c wdoc -n '__fish_using_command wdoc' -f -a md-printer
complete -c wdoc -n '__fish_using_command wdoc' -f -a query-task
complete -c wdoc -n '__fish_using_command wdoc' -f -a parse-file
complete -c wdoc -n '__fish_using_command wdoc' -f -a summary-task
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command wdoc; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a youtube-translation
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a out-file
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a whisper-lang
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a source-tag
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a anki-template
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a youtube-audio-backend
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a filter-content
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a audio-unsilence
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a doccheck-min-token
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a path
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a whisper-prompt
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a anki-tag-filter
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a loading-failure
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a anki-deck
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a online-media-url-regex
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a youtube-language
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a anki-profile
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a load-functions
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a embed-instruct
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a anki-notetype
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a doccheck-max-token
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a exclude
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a audio-backend
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a json-dict-template
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a pdf-parsers
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a anki-tag-render-filter
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a deepgram-kwargs
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a include
complete -c wdoc -n '__fish_using_command allowed-extra-args' -f -a filter-metadata
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command allowed-extra-args; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isidentifier; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --object' -l object
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command append; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command hex; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --default' -l default
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --key' -l key
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command get; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command source-tag' -f -a isnumeric
complete -c wdoc -n '__fish_using_command source-tag' -f -a join
complete -c wdoc -n '__fish_using_command source-tag' -f -a lstrip
complete -c wdoc -n '__fish_using_command source-tag' -f -a islower
complete -c wdoc -n '__fish_using_command source-tag' -f -a removeprefix
complete -c wdoc -n '__fish_using_command source-tag' -f -a rfind
complete -c wdoc -n '__fish_using_command source-tag' -f -a format-map
complete -c wdoc -n '__fish_using_command source-tag' -f -a isidentifier
complete -c wdoc -n '__fish_using_command source-tag' -f -a upper
complete -c wdoc -n '__fish_using_command source-tag' -f -a istitle
complete -c wdoc -n '__fish_using_command source-tag' -f -a isalnum
complete -c wdoc -n '__fish_using_command source-tag' -f -a startswith
complete -c wdoc -n '__fish_using_command source-tag' -f -a expandtabs
complete -c wdoc -n '__fish_using_command source-tag' -f -a ljust
complete -c wdoc -n '__fish_using_command source-tag' -f -a splitlines
complete -c wdoc -n '__fish_using_command source-tag' -f -a isspace
complete -c wdoc -n '__fish_using_command source-tag' -f -a rpartition
complete -c wdoc -n '__fish_using_command source-tag' -f -a title
complete -c wdoc -n '__fish_using_command source-tag' -f -a replace
complete -c wdoc -n '__fish_using_command source-tag' -f -a rjust
complete -c wdoc -n '__fish_using_command source-tag' -f -a format
complete -c wdoc -n '__fish_using_command source-tag' -f -a swapcase
complete -c wdoc -n '__fish_using_command source-tag' -f -a capitalize
complete -c wdoc -n '__fish_using_command source-tag' -f -a isalpha
complete -c wdoc -n '__fish_using_command source-tag' -f -a isupper
complete -c wdoc -n '__fish_using_command source-tag' -f -a zfill
complete -c wdoc -n '__fish_using_command source-tag' -f -a isdecimal
complete -c wdoc -n '__fish_using_command source-tag' -f -a lower
complete -c wdoc -n '__fish_using_command source-tag' -f -a rindex
complete -c wdoc -n '__fish_using_command source-tag' -f -a isascii
complete -c wdoc -n '__fish_using_command source-tag' -f -a casefold
complete -c wdoc -n '__fish_using_command source-tag' -f -a endswith
complete -c wdoc -n '__fish_using_command source-tag' -f -a strip
complete -c wdoc -n '__fish_using_command source-tag' -f -a isdigit
complete -c wdoc -n '__fish_using_command source-tag' -f -a count
complete -c wdoc -n '__fish_using_command source-tag' -f -a rsplit
complete -c wdoc -n '__fish_using_command source-tag' -f -a removesuffix
complete -c wdoc -n '__fish_using_command source-tag' -f -a rstrip
complete -c wdoc -n '__fish_using_command source-tag' -f -a partition
complete -c wdoc -n '__fish_using_command source-tag' -f -a translate
complete -c wdoc -n '__fish_using_command source-tag' -f -a encode
complete -c wdoc -n '__fish_using_command source-tag' -f -a find
complete -c wdoc -n '__fish_using_command source-tag' -f -a isprintable
complete -c wdoc -n '__fish_using_command source-tag' -f -a split
complete -c wdoc -n '__fish_using_command source-tag' -f -a center
complete -c wdoc -n '__fish_using_command source-tag' -f -a index
complete -c wdoc -n '__fish_using_command source-tag' -f -a maketrans
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command source-tag; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command istitle; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isalnum; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command anki-template' -f -a isnumeric
complete -c wdoc -n '__fish_using_command anki-template' -f -a join
complete -c wdoc -n '__fish_using_command anki-template' -f -a lstrip
complete -c wdoc -n '__fish_using_command anki-template' -f -a islower
complete -c wdoc -n '__fish_using_command anki-template' -f -a removeprefix
complete -c wdoc -n '__fish_using_command anki-template' -f -a rfind
complete -c wdoc -n '__fish_using_command anki-template' -f -a format-map
complete -c wdoc -n '__fish_using_command anki-template' -f -a isidentifier
complete -c wdoc -n '__fish_using_command anki-template' -f -a upper
complete -c wdoc -n '__fish_using_command anki-template' -f -a istitle
complete -c wdoc -n '__fish_using_command anki-template' -f -a isalnum
complete -c wdoc -n '__fish_using_command anki-template' -f -a startswith
complete -c wdoc -n '__fish_using_command anki-template' -f -a expandtabs
complete -c wdoc -n '__fish_using_command anki-template' -f -a ljust
complete -c wdoc -n '__fish_using_command anki-template' -f -a splitlines
complete -c wdoc -n '__fish_using_command anki-template' -f -a isspace
complete -c wdoc -n '__fish_using_command anki-template' -f -a rpartition
complete -c wdoc -n '__fish_using_command anki-template' -f -a title
complete -c wdoc -n '__fish_using_command anki-template' -f -a replace
complete -c wdoc -n '__fish_using_command anki-template' -f -a rjust
complete -c wdoc -n '__fish_using_command anki-template' -f -a format
complete -c wdoc -n '__fish_using_command anki-template' -f -a swapcase
complete -c wdoc -n '__fish_using_command anki-template' -f -a capitalize
complete -c wdoc -n '__fish_using_command anki-template' -f -a isalpha
complete -c wdoc -n '__fish_using_command anki-template' -f -a isupper
complete -c wdoc -n '__fish_using_command anki-template' -f -a zfill
complete -c wdoc -n '__fish_using_command anki-template' -f -a isdecimal
complete -c wdoc -n '__fish_using_command anki-template' -f -a lower
complete -c wdoc -n '__fish_using_command anki-template' -f -a rindex
complete -c wdoc -n '__fish_using_command anki-template' -f -a isascii
complete -c wdoc -n '__fish_using_command anki-template' -f -a casefold
complete -c wdoc -n '__fish_using_command anki-template' -f -a endswith
complete -c wdoc -n '__fish_using_command anki-template' -f -a strip
complete -c wdoc -n '__fish_using_command anki-template' -f -a isdigit
complete -c wdoc -n '__fish_using_command anki-template' -f -a count
complete -c wdoc -n '__fish_using_command anki-template' -f -a rsplit
complete -c wdoc -n '__fish_using_command anki-template' -f -a removesuffix
complete -c wdoc -n '__fish_using_command anki-template' -f -a rstrip
complete -c wdoc -n '__fish_using_command anki-template' -f -a partition
complete -c wdoc -n '__fish_using_command anki-template' -f -a translate
complete -c wdoc -n '__fish_using_command anki-template' -f -a encode
complete -c wdoc -n '__fish_using_command anki-template' -f -a find
complete -c wdoc -n '__fish_using_command anki-template' -f -a isprintable
complete -c wdoc -n '__fish_using_command anki-template' -f -a split
complete -c wdoc -n '__fish_using_command anki-template' -f -a center
complete -c wdoc -n '__fish_using_command anki-template' -f -a index
complete -c wdoc -n '__fish_using_command anki-template' -f -a maketrans
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-template; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --tabsize' -l tabsize
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command expandtabs; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command youtube-audio-backend' -f -a copy-with
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command youtube-audio-backend; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --width' -l width
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --fillchar' -l fillchar
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command ljust; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --only-text' -l only-text
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --cli-kwargs' -l cli-kwargs
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command parse-file; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --signed' -l signed
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --bytes' -l bytes
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --byteorder' -l byteorder
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command from-bytes; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command filter-content' -f -a copy-with
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command filter-content; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command splitlines; and __is_prev_global; and __option_entered_check --keepends' -l keepends
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a imag
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a as-integer-ratio
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a denominator
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a to-bytes
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a conjugate
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a real
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a bit-length
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a numerator
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a bit-count
complete -c wdoc -n '__fish_using_command doccheck-min-token' -f -a from-bytes
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command doccheck-min-token; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --signed' -l signed
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --length' -l length
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --byteorder' -l byteorder
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command to-bytes; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --width' -l width
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --fillchar' -l fillchar
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command rjust; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isnumeric
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a join
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a lstrip
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a islower
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a removeprefix
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a rfind
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a format-map
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isidentifier
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a upper
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a istitle
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isalnum
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a startswith
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a expandtabs
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a ljust
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a splitlines
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isspace
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a rpartition
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a title
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a replace
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a rjust
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a format
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a swapcase
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a capitalize
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isalpha
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isupper
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a zfill
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isdecimal
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a lower
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a rindex
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isascii
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a casefold
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a endswith
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a strip
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isdigit
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a count
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a rsplit
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a removesuffix
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a rstrip
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a partition
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a translate
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a encode
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a find
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a isprintable
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a split
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a center
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a index
complete -c wdoc -n '__fish_using_command anki-tag-filter' -f -a maketrans
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-tag-filter; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command swapcase; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isnumeric
complete -c wdoc -n '__fish_using_command loading-failure' -f -a join
complete -c wdoc -n '__fish_using_command loading-failure' -f -a lstrip
complete -c wdoc -n '__fish_using_command loading-failure' -f -a islower
complete -c wdoc -n '__fish_using_command loading-failure' -f -a removeprefix
complete -c wdoc -n '__fish_using_command loading-failure' -f -a rfind
complete -c wdoc -n '__fish_using_command loading-failure' -f -a format-map
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isidentifier
complete -c wdoc -n '__fish_using_command loading-failure' -f -a upper
complete -c wdoc -n '__fish_using_command loading-failure' -f -a istitle
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isalnum
complete -c wdoc -n '__fish_using_command loading-failure' -f -a startswith
complete -c wdoc -n '__fish_using_command loading-failure' -f -a expandtabs
complete -c wdoc -n '__fish_using_command loading-failure' -f -a ljust
complete -c wdoc -n '__fish_using_command loading-failure' -f -a splitlines
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isspace
complete -c wdoc -n '__fish_using_command loading-failure' -f -a rpartition
complete -c wdoc -n '__fish_using_command loading-failure' -f -a title
complete -c wdoc -n '__fish_using_command loading-failure' -f -a replace
complete -c wdoc -n '__fish_using_command loading-failure' -f -a rjust
complete -c wdoc -n '__fish_using_command loading-failure' -f -a format
complete -c wdoc -n '__fish_using_command loading-failure' -f -a swapcase
complete -c wdoc -n '__fish_using_command loading-failure' -f -a capitalize
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isalpha
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isupper
complete -c wdoc -n '__fish_using_command loading-failure' -f -a zfill
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isdecimal
complete -c wdoc -n '__fish_using_command loading-failure' -f -a lower
complete -c wdoc -n '__fish_using_command loading-failure' -f -a rindex
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isascii
complete -c wdoc -n '__fish_using_command loading-failure' -f -a casefold
complete -c wdoc -n '__fish_using_command loading-failure' -f -a endswith
complete -c wdoc -n '__fish_using_command loading-failure' -f -a strip
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isdigit
complete -c wdoc -n '__fish_using_command loading-failure' -f -a count
complete -c wdoc -n '__fish_using_command loading-failure' -f -a rsplit
complete -c wdoc -n '__fish_using_command loading-failure' -f -a removesuffix
complete -c wdoc -n '__fish_using_command loading-failure' -f -a rstrip
complete -c wdoc -n '__fish_using_command loading-failure' -f -a partition
complete -c wdoc -n '__fish_using_command loading-failure' -f -a translate
complete -c wdoc -n '__fish_using_command loading-failure' -f -a encode
complete -c wdoc -n '__fish_using_command loading-failure' -f -a find
complete -c wdoc -n '__fish_using_command loading-failure' -f -a isprintable
complete -c wdoc -n '__fish_using_command loading-failure' -f -a split
complete -c wdoc -n '__fish_using_command loading-failure' -f -a center
complete -c wdoc -n '__fish_using_command loading-failure' -f -a index
complete -c wdoc -n '__fish_using_command loading-failure' -f -a maketrans
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command loading-failure; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isupper; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --value' -l value
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command remove; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command prepare-query-task; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isnumeric
complete -c wdoc -n '__fish_using_command anki-deck' -f -a join
complete -c wdoc -n '__fish_using_command anki-deck' -f -a lstrip
complete -c wdoc -n '__fish_using_command anki-deck' -f -a islower
complete -c wdoc -n '__fish_using_command anki-deck' -f -a removeprefix
complete -c wdoc -n '__fish_using_command anki-deck' -f -a rfind
complete -c wdoc -n '__fish_using_command anki-deck' -f -a format-map
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isidentifier
complete -c wdoc -n '__fish_using_command anki-deck' -f -a upper
complete -c wdoc -n '__fish_using_command anki-deck' -f -a istitle
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isalnum
complete -c wdoc -n '__fish_using_command anki-deck' -f -a startswith
complete -c wdoc -n '__fish_using_command anki-deck' -f -a expandtabs
complete -c wdoc -n '__fish_using_command anki-deck' -f -a ljust
complete -c wdoc -n '__fish_using_command anki-deck' -f -a splitlines
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isspace
complete -c wdoc -n '__fish_using_command anki-deck' -f -a rpartition
complete -c wdoc -n '__fish_using_command anki-deck' -f -a title
complete -c wdoc -n '__fish_using_command anki-deck' -f -a replace
complete -c wdoc -n '__fish_using_command anki-deck' -f -a rjust
complete -c wdoc -n '__fish_using_command anki-deck' -f -a format
complete -c wdoc -n '__fish_using_command anki-deck' -f -a swapcase
complete -c wdoc -n '__fish_using_command anki-deck' -f -a capitalize
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isalpha
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isupper
complete -c wdoc -n '__fish_using_command anki-deck' -f -a zfill
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isdecimal
complete -c wdoc -n '__fish_using_command anki-deck' -f -a lower
complete -c wdoc -n '__fish_using_command anki-deck' -f -a rindex
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isascii
complete -c wdoc -n '__fish_using_command anki-deck' -f -a casefold
complete -c wdoc -n '__fish_using_command anki-deck' -f -a endswith
complete -c wdoc -n '__fish_using_command anki-deck' -f -a strip
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isdigit
complete -c wdoc -n '__fish_using_command anki-deck' -f -a count
complete -c wdoc -n '__fish_using_command anki-deck' -f -a rsplit
complete -c wdoc -n '__fish_using_command anki-deck' -f -a removesuffix
complete -c wdoc -n '__fish_using_command anki-deck' -f -a rstrip
complete -c wdoc -n '__fish_using_command anki-deck' -f -a partition
complete -c wdoc -n '__fish_using_command anki-deck' -f -a translate
complete -c wdoc -n '__fish_using_command anki-deck' -f -a encode
complete -c wdoc -n '__fish_using_command anki-deck' -f -a find
complete -c wdoc -n '__fish_using_command anki-deck' -f -a isprintable
complete -c wdoc -n '__fish_using_command anki-deck' -f -a split
complete -c wdoc -n '__fish_using_command anki-deck' -f -a center
complete -c wdoc -n '__fish_using_command anki-deck' -f -a index
complete -c wdoc -n '__fish_using_command anki-deck' -f -a maketrans
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-deck; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isnumeric
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a join
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a lstrip
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a islower
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a removeprefix
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a rfind
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a format-map
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isidentifier
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a upper
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a istitle
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isalnum
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a startswith
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a expandtabs
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a ljust
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a splitlines
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isspace
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a rpartition
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a title
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a replace
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a rjust
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a format
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a swapcase
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a capitalize
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isalpha
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isupper
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a zfill
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isdecimal
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a lower
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a rindex
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isascii
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a casefold
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a endswith
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a strip
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isdigit
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a count
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a rsplit
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a removesuffix
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a rstrip
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a partition
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a translate
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a encode
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a find
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a isprintable
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a split
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a center
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a index
complete -c wdoc -n '__fish_using_command online-media-url-regex' -f -a maketrans
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command online-media-url-regex; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isnumeric
complete -c wdoc -n '__fish_using_command youtube-language' -f -a join
complete -c wdoc -n '__fish_using_command youtube-language' -f -a lstrip
complete -c wdoc -n '__fish_using_command youtube-language' -f -a islower
complete -c wdoc -n '__fish_using_command youtube-language' -f -a removeprefix
complete -c wdoc -n '__fish_using_command youtube-language' -f -a rfind
complete -c wdoc -n '__fish_using_command youtube-language' -f -a format-map
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isidentifier
complete -c wdoc -n '__fish_using_command youtube-language' -f -a upper
complete -c wdoc -n '__fish_using_command youtube-language' -f -a istitle
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isalnum
complete -c wdoc -n '__fish_using_command youtube-language' -f -a startswith
complete -c wdoc -n '__fish_using_command youtube-language' -f -a expandtabs
complete -c wdoc -n '__fish_using_command youtube-language' -f -a ljust
complete -c wdoc -n '__fish_using_command youtube-language' -f -a splitlines
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isspace
complete -c wdoc -n '__fish_using_command youtube-language' -f -a rpartition
complete -c wdoc -n '__fish_using_command youtube-language' -f -a title
complete -c wdoc -n '__fish_using_command youtube-language' -f -a replace
complete -c wdoc -n '__fish_using_command youtube-language' -f -a rjust
complete -c wdoc -n '__fish_using_command youtube-language' -f -a format
complete -c wdoc -n '__fish_using_command youtube-language' -f -a swapcase
complete -c wdoc -n '__fish_using_command youtube-language' -f -a capitalize
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isalpha
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isupper
complete -c wdoc -n '__fish_using_command youtube-language' -f -a zfill
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isdecimal
complete -c wdoc -n '__fish_using_command youtube-language' -f -a lower
complete -c wdoc -n '__fish_using_command youtube-language' -f -a rindex
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isascii
complete -c wdoc -n '__fish_using_command youtube-language' -f -a casefold
complete -c wdoc -n '__fish_using_command youtube-language' -f -a endswith
complete -c wdoc -n '__fish_using_command youtube-language' -f -a strip
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isdigit
complete -c wdoc -n '__fish_using_command youtube-language' -f -a count
complete -c wdoc -n '__fish_using_command youtube-language' -f -a rsplit
complete -c wdoc -n '__fish_using_command youtube-language' -f -a removesuffix
complete -c wdoc -n '__fish_using_command youtube-language' -f -a rstrip
complete -c wdoc -n '__fish_using_command youtube-language' -f -a partition
complete -c wdoc -n '__fish_using_command youtube-language' -f -a translate
complete -c wdoc -n '__fish_using_command youtube-language' -f -a encode
complete -c wdoc -n '__fish_using_command youtube-language' -f -a find
complete -c wdoc -n '__fish_using_command youtube-language' -f -a isprintable
complete -c wdoc -n '__fish_using_command youtube-language' -f -a split
complete -c wdoc -n '__fish_using_command youtube-language' -f -a center
complete -c wdoc -n '__fish_using_command youtube-language' -f -a index
complete -c wdoc -n '__fish_using_command youtube-language' -f -a maketrans
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command youtube-language; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command is-integer; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --width' -l width
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command zfill; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isnumeric
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a join
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a lstrip
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a islower
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a removeprefix
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a rfind
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a format-map
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isidentifier
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a upper
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a istitle
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isalnum
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a startswith
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a expandtabs
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a ljust
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a splitlines
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isspace
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a rpartition
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a title
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a replace
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a rjust
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a format
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a swapcase
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a capitalize
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isalpha
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isupper
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a zfill
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isdecimal
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a lower
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a rindex
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isascii
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a casefold
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a endswith
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a strip
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isdigit
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a count
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a rsplit
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a removesuffix
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a rstrip
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a partition
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a translate
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a encode
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a find
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a isprintable
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a split
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a center
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a index
complete -c wdoc -n '__fish_using_command anki-notetype' -f -a maketrans
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-notetype; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a imag
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a as-integer-ratio
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a denominator
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a to-bytes
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a conjugate
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a real
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a bit-length
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a numerator
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a bit-count
complete -c wdoc -n '__fish_using_command doccheck-max-token' -f -a from-bytes
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command doccheck-max-token; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --chars' -l chars
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command strip; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command exclude' -f -a isnumeric
complete -c wdoc -n '__fish_using_command exclude' -f -a join
complete -c wdoc -n '__fish_using_command exclude' -f -a lstrip
complete -c wdoc -n '__fish_using_command exclude' -f -a islower
complete -c wdoc -n '__fish_using_command exclude' -f -a removeprefix
complete -c wdoc -n '__fish_using_command exclude' -f -a rfind
complete -c wdoc -n '__fish_using_command exclude' -f -a format-map
complete -c wdoc -n '__fish_using_command exclude' -f -a isidentifier
complete -c wdoc -n '__fish_using_command exclude' -f -a upper
complete -c wdoc -n '__fish_using_command exclude' -f -a istitle
complete -c wdoc -n '__fish_using_command exclude' -f -a isalnum
complete -c wdoc -n '__fish_using_command exclude' -f -a startswith
complete -c wdoc -n '__fish_using_command exclude' -f -a expandtabs
complete -c wdoc -n '__fish_using_command exclude' -f -a ljust
complete -c wdoc -n '__fish_using_command exclude' -f -a splitlines
complete -c wdoc -n '__fish_using_command exclude' -f -a isspace
complete -c wdoc -n '__fish_using_command exclude' -f -a rpartition
complete -c wdoc -n '__fish_using_command exclude' -f -a title
complete -c wdoc -n '__fish_using_command exclude' -f -a replace
complete -c wdoc -n '__fish_using_command exclude' -f -a rjust
complete -c wdoc -n '__fish_using_command exclude' -f -a format
complete -c wdoc -n '__fish_using_command exclude' -f -a swapcase
complete -c wdoc -n '__fish_using_command exclude' -f -a capitalize
complete -c wdoc -n '__fish_using_command exclude' -f -a isalpha
complete -c wdoc -n '__fish_using_command exclude' -f -a isupper
complete -c wdoc -n '__fish_using_command exclude' -f -a zfill
complete -c wdoc -n '__fish_using_command exclude' -f -a isdecimal
complete -c wdoc -n '__fish_using_command exclude' -f -a lower
complete -c wdoc -n '__fish_using_command exclude' -f -a rindex
complete -c wdoc -n '__fish_using_command exclude' -f -a isascii
complete -c wdoc -n '__fish_using_command exclude' -f -a casefold
complete -c wdoc -n '__fish_using_command exclude' -f -a endswith
complete -c wdoc -n '__fish_using_command exclude' -f -a strip
complete -c wdoc -n '__fish_using_command exclude' -f -a isdigit
complete -c wdoc -n '__fish_using_command exclude' -f -a count
complete -c wdoc -n '__fish_using_command exclude' -f -a rsplit
complete -c wdoc -n '__fish_using_command exclude' -f -a removesuffix
complete -c wdoc -n '__fish_using_command exclude' -f -a rstrip
complete -c wdoc -n '__fish_using_command exclude' -f -a partition
complete -c wdoc -n '__fish_using_command exclude' -f -a translate
complete -c wdoc -n '__fish_using_command exclude' -f -a encode
complete -c wdoc -n '__fish_using_command exclude' -f -a find
complete -c wdoc -n '__fish_using_command exclude' -f -a isprintable
complete -c wdoc -n '__fish_using_command exclude' -f -a split
complete -c wdoc -n '__fish_using_command exclude' -f -a center
complete -c wdoc -n '__fish_using_command exclude' -f -a index
complete -c wdoc -n '__fish_using_command exclude' -f -a maketrans
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command exclude; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isdigit; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --default' -l default
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --key' -l key
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command setdefault; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --value' -l value
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command count; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command audio-backend' -f -a copy-with
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command audio-backend; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --sep' -l sep
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --maxsplit' -l maxsplit
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command rsplit; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command copy; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command bit-count; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --index' -l index
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command pop; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isnumeric
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a join
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a lstrip
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a islower
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a removeprefix
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a rfind
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a format-map
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isidentifier
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a upper
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a istitle
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isalnum
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a startswith
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a expandtabs
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a ljust
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a splitlines
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isspace
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a rpartition
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a title
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a replace
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a rjust
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a format
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a swapcase
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a capitalize
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isalpha
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isupper
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a zfill
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isdecimal
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a lower
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a rindex
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isascii
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a casefold
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a endswith
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a strip
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isdigit
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a count
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a rsplit
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a removesuffix
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a rstrip
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a partition
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a translate
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a encode
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a find
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a isprintable
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a split
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a center
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a index
complete -c wdoc -n '__fish_using_command json-dict-template' -f -a maketrans
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command json-dict-template; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --string' -l string
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command fromhex; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --reverse' -l reverse
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --key' -l key
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command sort; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command pdf-parsers' -f -a copy-with
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command pdf-parsers; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --sep' -l sep
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command partition; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isnumeric
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a join
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a lstrip
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a islower
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a removeprefix
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a rfind
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a format-map
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isidentifier
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a upper
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a istitle
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isalnum
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a startswith
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a expandtabs
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a ljust
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a splitlines
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isspace
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a rpartition
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a title
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a replace
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a rjust
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a format
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a swapcase
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a capitalize
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isalpha
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isupper
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a zfill
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isdecimal
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a lower
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a rindex
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isascii
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a casefold
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a endswith
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a strip
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isdigit
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a count
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a rsplit
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a removesuffix
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a rstrip
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a partition
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a translate
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a encode
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a find
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a isprintable
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a split
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a center
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a index
complete -c wdoc -n '__fish_using_command anki-tag-render-filter' -f -a maketrans
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-tag-render-filter; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --encoding' -l encoding
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --errors' -l errors
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command encode; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isprintable; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command VERSION' -f -a isnumeric
complete -c wdoc -n '__fish_using_command VERSION' -f -a join
complete -c wdoc -n '__fish_using_command VERSION' -f -a lstrip
complete -c wdoc -n '__fish_using_command VERSION' -f -a islower
complete -c wdoc -n '__fish_using_command VERSION' -f -a removeprefix
complete -c wdoc -n '__fish_using_command VERSION' -f -a rfind
complete -c wdoc -n '__fish_using_command VERSION' -f -a format-map
complete -c wdoc -n '__fish_using_command VERSION' -f -a isidentifier
complete -c wdoc -n '__fish_using_command VERSION' -f -a upper
complete -c wdoc -n '__fish_using_command VERSION' -f -a istitle
complete -c wdoc -n '__fish_using_command VERSION' -f -a isalnum
complete -c wdoc -n '__fish_using_command VERSION' -f -a startswith
complete -c wdoc -n '__fish_using_command VERSION' -f -a expandtabs
complete -c wdoc -n '__fish_using_command VERSION' -f -a ljust
complete -c wdoc -n '__fish_using_command VERSION' -f -a splitlines
complete -c wdoc -n '__fish_using_command VERSION' -f -a isspace
complete -c wdoc -n '__fish_using_command VERSION' -f -a rpartition
complete -c wdoc -n '__fish_using_command VERSION' -f -a title
complete -c wdoc -n '__fish_using_command VERSION' -f -a replace
complete -c wdoc -n '__fish_using_command VERSION' -f -a rjust
complete -c wdoc -n '__fish_using_command VERSION' -f -a format
complete -c wdoc -n '__fish_using_command VERSION' -f -a swapcase
complete -c wdoc -n '__fish_using_command VERSION' -f -a capitalize
complete -c wdoc -n '__fish_using_command VERSION' -f -a isalpha
complete -c wdoc -n '__fish_using_command VERSION' -f -a isupper
complete -c wdoc -n '__fish_using_command VERSION' -f -a zfill
complete -c wdoc -n '__fish_using_command VERSION' -f -a isdecimal
complete -c wdoc -n '__fish_using_command VERSION' -f -a lower
complete -c wdoc -n '__fish_using_command VERSION' -f -a rindex
complete -c wdoc -n '__fish_using_command VERSION' -f -a isascii
complete -c wdoc -n '__fish_using_command VERSION' -f -a casefold
complete -c wdoc -n '__fish_using_command VERSION' -f -a endswith
complete -c wdoc -n '__fish_using_command VERSION' -f -a strip
complete -c wdoc -n '__fish_using_command VERSION' -f -a isdigit
complete -c wdoc -n '__fish_using_command VERSION' -f -a count
complete -c wdoc -n '__fish_using_command VERSION' -f -a rsplit
complete -c wdoc -n '__fish_using_command VERSION' -f -a removesuffix
complete -c wdoc -n '__fish_using_command VERSION' -f -a rstrip
complete -c wdoc -n '__fish_using_command VERSION' -f -a partition
complete -c wdoc -n '__fish_using_command VERSION' -f -a translate
complete -c wdoc -n '__fish_using_command VERSION' -f -a encode
complete -c wdoc -n '__fish_using_command VERSION' -f -a find
complete -c wdoc -n '__fish_using_command VERSION' -f -a isprintable
complete -c wdoc -n '__fish_using_command VERSION' -f -a split
complete -c wdoc -n '__fish_using_command VERSION' -f -a center
complete -c wdoc -n '__fish_using_command VERSION' -f -a index
complete -c wdoc -n '__fish_using_command VERSION' -f -a maketrans
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command VERSION; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command include' -f -a isnumeric
complete -c wdoc -n '__fish_using_command include' -f -a join
complete -c wdoc -n '__fish_using_command include' -f -a lstrip
complete -c wdoc -n '__fish_using_command include' -f -a islower
complete -c wdoc -n '__fish_using_command include' -f -a removeprefix
complete -c wdoc -n '__fish_using_command include' -f -a rfind
complete -c wdoc -n '__fish_using_command include' -f -a format-map
complete -c wdoc -n '__fish_using_command include' -f -a isidentifier
complete -c wdoc -n '__fish_using_command include' -f -a upper
complete -c wdoc -n '__fish_using_command include' -f -a istitle
complete -c wdoc -n '__fish_using_command include' -f -a isalnum
complete -c wdoc -n '__fish_using_command include' -f -a startswith
complete -c wdoc -n '__fish_using_command include' -f -a expandtabs
complete -c wdoc -n '__fish_using_command include' -f -a ljust
complete -c wdoc -n '__fish_using_command include' -f -a splitlines
complete -c wdoc -n '__fish_using_command include' -f -a isspace
complete -c wdoc -n '__fish_using_command include' -f -a rpartition
complete -c wdoc -n '__fish_using_command include' -f -a title
complete -c wdoc -n '__fish_using_command include' -f -a replace
complete -c wdoc -n '__fish_using_command include' -f -a rjust
complete -c wdoc -n '__fish_using_command include' -f -a format
complete -c wdoc -n '__fish_using_command include' -f -a swapcase
complete -c wdoc -n '__fish_using_command include' -f -a capitalize
complete -c wdoc -n '__fish_using_command include' -f -a isalpha
complete -c wdoc -n '__fish_using_command include' -f -a isupper
complete -c wdoc -n '__fish_using_command include' -f -a zfill
complete -c wdoc -n '__fish_using_command include' -f -a isdecimal
complete -c wdoc -n '__fish_using_command include' -f -a lower
complete -c wdoc -n '__fish_using_command include' -f -a rindex
complete -c wdoc -n '__fish_using_command include' -f -a isascii
complete -c wdoc -n '__fish_using_command include' -f -a casefold
complete -c wdoc -n '__fish_using_command include' -f -a endswith
complete -c wdoc -n '__fish_using_command include' -f -a strip
complete -c wdoc -n '__fish_using_command include' -f -a isdigit
complete -c wdoc -n '__fish_using_command include' -f -a count
complete -c wdoc -n '__fish_using_command include' -f -a rsplit
complete -c wdoc -n '__fish_using_command include' -f -a removesuffix
complete -c wdoc -n '__fish_using_command include' -f -a rstrip
complete -c wdoc -n '__fish_using_command include' -f -a partition
complete -c wdoc -n '__fish_using_command include' -f -a translate
complete -c wdoc -n '__fish_using_command include' -f -a encode
complete -c wdoc -n '__fish_using_command include' -f -a find
complete -c wdoc -n '__fish_using_command include' -f -a isprintable
complete -c wdoc -n '__fish_using_command include' -f -a split
complete -c wdoc -n '__fish_using_command include' -f -a center
complete -c wdoc -n '__fish_using_command include' -f -a index
complete -c wdoc -n '__fish_using_command include' -f -a maketrans
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command include; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --sep' -l sep
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --maxsplit' -l maxsplit
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command split; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --color' -l color
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --message' -l message
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command md-printer; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isspace; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command filter-metadata' -f -a copy-with
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command filter-metadata; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --start' -l start
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --stop' -l stop
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --value' -l value
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command index; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isnumeric; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --iterable' -l iterable
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command join; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isnumeric
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a join
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a lstrip
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a islower
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a removeprefix
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a rfind
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a format-map
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isidentifier
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a upper
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a istitle
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isalnum
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a startswith
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a expandtabs
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a ljust
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a splitlines
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isspace
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a rpartition
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a title
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a replace
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a rjust
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a format
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a swapcase
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a capitalize
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isalpha
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isupper
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a zfill
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isdecimal
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a lower
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a rindex
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isascii
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a casefold
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a endswith
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a strip
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isdigit
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a count
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a rsplit
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a removesuffix
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a rstrip
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a partition
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a translate
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a encode
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a find
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a isprintable
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a split
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a center
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a index
complete -c wdoc -n '__fish_using_command youtube-translation' -f -a maketrans
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command youtube-translation; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command out-file' -f -a copy-with
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command out-file; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isnumeric
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a join
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a lstrip
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a islower
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a removeprefix
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a rfind
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a format-map
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isidentifier
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a upper
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a istitle
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isalnum
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a startswith
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a expandtabs
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a ljust
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a splitlines
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isspace
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a rpartition
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a title
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a replace
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a rjust
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a format
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a swapcase
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a capitalize
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isalpha
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isupper
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a zfill
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isdecimal
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a lower
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a rindex
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isascii
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a casefold
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a endswith
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a strip
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isdigit
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a count
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a rsplit
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a removesuffix
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a rstrip
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a partition
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a translate
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a encode
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a find
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a isprintable
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a split
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a center
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a index
complete -c wdoc -n '__fish_using_command whisper-lang' -f -a maketrans
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command whisper-lang; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --chars' -l chars
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command lstrip; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command islower; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command popitem; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --params' -l params
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --args' -l args
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command copy-with; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a imag
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a as-integer-ratio
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a conjugate
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a real
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a hex
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a fromhex
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob' -f -a is-integer
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --x' -l x
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command doccheck-min-lang-prob; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --iterable' -l iterable
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command extend; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command query-task; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a imag
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a as-integer-ratio
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a denominator
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a to-bytes
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a conjugate
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a real
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a bit-length
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a numerator
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a bit-count
complete -c wdoc -n '__fish_using_command audio-unsilence' -f -a from-bytes
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command audio-unsilence; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --sep' -l sep
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command rpartition; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command path' -f -a copy-with
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command path; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command title; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --new' -l new
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --count' -l count
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --old' -l old
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command replace; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isnumeric
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a join
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a lstrip
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a islower
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a removeprefix
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a rfind
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a format-map
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isidentifier
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a upper
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a istitle
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isalnum
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a startswith
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a expandtabs
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a ljust
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a splitlines
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isspace
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a rpartition
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a title
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a replace
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a rjust
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a format
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a swapcase
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a capitalize
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isalpha
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isupper
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a zfill
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isdecimal
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a lower
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a rindex
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isascii
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a casefold
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a endswith
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a strip
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isdigit
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a count
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a rsplit
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a removesuffix
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a rstrip
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a partition
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a translate
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a encode
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a find
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a isprintable
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a split
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a center
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a index
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex' -f -a maketrans
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command online-media-resourcetype-regex; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isnumeric
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a join
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a lstrip
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a islower
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a removeprefix
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a rfind
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a format-map
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isidentifier
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a upper
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a istitle
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isalnum
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a startswith
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a expandtabs
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a ljust
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a splitlines
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isspace
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a rpartition
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a title
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a replace
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a rjust
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a format
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a swapcase
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a capitalize
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isalpha
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isupper
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a zfill
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isdecimal
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a lower
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a rindex
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isascii
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a casefold
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a endswith
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a strip
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isdigit
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a count
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a rsplit
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a removesuffix
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a rstrip
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a partition
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a translate
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a encode
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a find
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a isprintable
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a split
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a center
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a index
complete -c wdoc -n '__fish_using_command whisper-prompt' -f -a maketrans
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command whisper-prompt; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --object' -l object
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --index' -l index
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command insert; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command capitalize; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isalpha; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command summary-task; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isdecimal; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command lower; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command as-integer-ratio; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isnumeric
complete -c wdoc -n '__fish_using_command anki-profile' -f -a join
complete -c wdoc -n '__fish_using_command anki-profile' -f -a lstrip
complete -c wdoc -n '__fish_using_command anki-profile' -f -a islower
complete -c wdoc -n '__fish_using_command anki-profile' -f -a removeprefix
complete -c wdoc -n '__fish_using_command anki-profile' -f -a rfind
complete -c wdoc -n '__fish_using_command anki-profile' -f -a format-map
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isidentifier
complete -c wdoc -n '__fish_using_command anki-profile' -f -a upper
complete -c wdoc -n '__fish_using_command anki-profile' -f -a istitle
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isalnum
complete -c wdoc -n '__fish_using_command anki-profile' -f -a startswith
complete -c wdoc -n '__fish_using_command anki-profile' -f -a expandtabs
complete -c wdoc -n '__fish_using_command anki-profile' -f -a ljust
complete -c wdoc -n '__fish_using_command anki-profile' -f -a splitlines
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isspace
complete -c wdoc -n '__fish_using_command anki-profile' -f -a rpartition
complete -c wdoc -n '__fish_using_command anki-profile' -f -a title
complete -c wdoc -n '__fish_using_command anki-profile' -f -a replace
complete -c wdoc -n '__fish_using_command anki-profile' -f -a rjust
complete -c wdoc -n '__fish_using_command anki-profile' -f -a format
complete -c wdoc -n '__fish_using_command anki-profile' -f -a swapcase
complete -c wdoc -n '__fish_using_command anki-profile' -f -a capitalize
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isalpha
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isupper
complete -c wdoc -n '__fish_using_command anki-profile' -f -a zfill
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isdecimal
complete -c wdoc -n '__fish_using_command anki-profile' -f -a lower
complete -c wdoc -n '__fish_using_command anki-profile' -f -a rindex
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isascii
complete -c wdoc -n '__fish_using_command anki-profile' -f -a casefold
complete -c wdoc -n '__fish_using_command anki-profile' -f -a endswith
complete -c wdoc -n '__fish_using_command anki-profile' -f -a strip
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isdigit
complete -c wdoc -n '__fish_using_command anki-profile' -f -a count
complete -c wdoc -n '__fish_using_command anki-profile' -f -a rsplit
complete -c wdoc -n '__fish_using_command anki-profile' -f -a removesuffix
complete -c wdoc -n '__fish_using_command anki-profile' -f -a rstrip
complete -c wdoc -n '__fish_using_command anki-profile' -f -a partition
complete -c wdoc -n '__fish_using_command anki-profile' -f -a translate
complete -c wdoc -n '__fish_using_command anki-profile' -f -a encode
complete -c wdoc -n '__fish_using_command anki-profile' -f -a find
complete -c wdoc -n '__fish_using_command anki-profile' -f -a isprintable
complete -c wdoc -n '__fish_using_command anki-profile' -f -a split
complete -c wdoc -n '__fish_using_command anki-profile' -f -a center
complete -c wdoc -n '__fish_using_command anki-profile' -f -a index
complete -c wdoc -n '__fish_using_command anki-profile' -f -a maketrans
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command anki-profile; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command load-functions' -f -a copy-with
complete -c wdoc -n '__fish_using_command load-functions' -f -a append
complete -c wdoc -n '__fish_using_command load-functions' -f -a extend
complete -c wdoc -n '__fish_using_command load-functions' -f -a insert
complete -c wdoc -n '__fish_using_command load-functions' -f -a reverse
complete -c wdoc -n '__fish_using_command load-functions' -f -a count
complete -c wdoc -n '__fish_using_command load-functions' -f -a remove
complete -c wdoc -n '__fish_using_command load-functions' -f -a copy
complete -c wdoc -n '__fish_using_command load-functions' -f -a pop
complete -c wdoc -n '__fish_using_command load-functions' -f -a clear
complete -c wdoc -n '__fish_using_command load-functions' -f -a index
complete -c wdoc -n '__fish_using_command load-functions' -f -a sort
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command load-functions; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command isascii; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isnumeric
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a join
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a lstrip
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a islower
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a removeprefix
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a rfind
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a format-map
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isidentifier
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a upper
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a istitle
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isalnum
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a startswith
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a expandtabs
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a ljust
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a splitlines
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isspace
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a rpartition
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a title
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a replace
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a rjust
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a format
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a swapcase
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a capitalize
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isalpha
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isupper
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a zfill
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isdecimal
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a lower
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a rindex
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isascii
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a casefold
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a endswith
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a strip
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isdigit
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a count
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a rsplit
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a removesuffix
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a rstrip
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a partition
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a translate
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a encode
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a find
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a isprintable
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a split
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a center
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a index
complete -c wdoc -n '__fish_using_command embed-instruct' -f -a maketrans
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command embed-instruct; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command casefold; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command bit-length; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --suffix' -l suffix
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command removesuffix; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command clear; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --chars' -l chars
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command rstrip; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --table' -l table
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command translate; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a copy-with
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a append
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a extend
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a insert
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a reverse
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a count
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a remove
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a copy
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a pop
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a clear
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a index
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys' -f -a sort
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command json-dict-exclude-keys; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a keys
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a popitem
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a fromkeys
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a setdefault
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a get
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a values
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a copy
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a items
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a pop
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a clear
complete -c wdoc -n '__fish_using_command deepgram-kwargs' -f -a update
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command deepgram-kwargs; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command conjugate; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --iterable' -l iterable
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --value' -l value
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command fromkeys; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command reverse; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --width' -l width
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --fillchar' -l fillchar
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command center; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --dollar-limit' -l dollar-limit
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --file-loader-parallel-backend' -l file-loader-parallel-backend
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --path' -l path
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --exclude' -l exclude
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --anki-template' -l anki-template
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --llm-verbosity' -l llm-verbosity
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --online-media-url-regex' -l online-media-url-regex
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --summary-n-recursion' -l summary-n-recursion
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --query-retrievers' -l query-retrievers
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --llms-api-bases' -l llms-api-bases
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --embed-model' -l embed-model
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --silent' -l silent
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --save-embeds-as' -l save-embeds-as
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --notification-callback' -l notification-callback
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --DIY-rolling-window-embedding' -l DIY-rolling-window-embedding
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --private' -l private
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --filter-metadata' -l filter-metadata
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --debug' -l debug
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --whisper-prompt' -l whisper-prompt
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --whisper-lang' -l whisper-lang
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --task' -l task
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --loading-failure' -l loading-failure
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --pdf-parsers' -l pdf-parsers
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --json-dict-template' -l json-dict-template
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --anki-notetype' -l anki-notetype
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --query-eval-check-number' -l query-eval-check-number
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --file-loader-n-jobs' -l file-loader-n-jobs
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --self' -l self
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --modelname' -l modelname
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --anki-profile' -l anki-profile
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --summary-language' -l summary-language
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --filter-content' -l filter-content
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --doccheck-min-lang-prob' -l doccheck-min-lang-prob
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --query-eval-modelname' -l query-eval-modelname
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --doccheck-max-token' -l doccheck-max-token
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --embed-kwargs' -l embed-kwargs
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --youtube-audio-backend' -l youtube-audio-backend
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --disable-md-printing' -l disable-md-printing
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --anki-deck' -l anki-deck
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --online-media-resourcetype-regex' -l online-media-resourcetype-regex
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --deepgram-kwargs' -l deepgram-kwargs
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --disable-llm-cache' -l disable-llm-cache
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --query' -l query
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --audio-backend' -l audio-backend
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --audio-unsilence' -l audio-unsilence
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --import-mode' -l import-mode
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --youtube-language' -l youtube-language
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --out-file' -l out-file
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --verbose' -l verbose
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --source-tag' -l source-tag
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --include' -l include
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --json-dict-exclude-keys' -l json-dict-exclude-keys
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --filetype' -l filetype
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --top-k' -l top-k
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --query-relevancy' -l query-relevancy
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --embed-instruct' -l embed-instruct
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --load-embeds-from' -l load-embeds-from
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --youtube-translation' -l youtube-translation
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --doccheck-min-token' -l doccheck-min-token
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --version' -l version
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --anki-tag-filter' -l anki-tag-filter
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --load-functions' -l load-functions
complete -c wdoc -n '__fish_using_command upper; and __is_prev_global; and __option_entered_check --anki-tag-render-filter' -l anki-tag-render-filter

