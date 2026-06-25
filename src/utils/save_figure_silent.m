function save_figure_silent(fig, filename)
    % Save figure with target filename under silent/visibility toggle settings
    set(fig, 'Visible', 'on');
    drawnow; pause(0.2);
    try
        print(fig, filename, '-dpng', '-r300');
    catch
        saveas(fig, filename);
    end
    set(fig, 'Visible', 'off');
end
